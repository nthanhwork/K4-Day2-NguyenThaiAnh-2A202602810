"""Report figures, kept separate from the immutable numeric inference protocol."""
from pathlib import Path

import matplotlib.pyplot as plt

import inference_notebook as runner


def plot_tradeoff(session_id='I_T04_seed0'):
    cfg = runner.source_context()[0]
    frame = runner.collect_session(session_id,cfg=cfg)
    if 'latency_p95_ms' not in frame:
        return None
    measured = frame[frame.status == 'complete']
    if measured.empty:
        return None
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    offsets = {'I00':(5,32),'I07':(55,12),'I08':(55,-12),'I01':(8,-3),'I02':(8,3)}
    for index,row in enumerate(measured.itertuples()):
        color = f'C{index}'
        for ax in axes:
            ax.scatter(row.latency_p95_ms,row.macro_f1*100,color=color,label=row.method,s=40)
        axes[0].annotate(row.method,(row.latency_p95_ms,row.macro_f1*100),
                         xytext=offsets[row.method],textcoords='offset points',color=color,
                         arrowprops={'arrowstyle':'-','color':color,'linewidth':.7})
    low,high = measured.macro_f1.min()*100, measured.macro_f1.max()*100
    pad = max((high-low)*.4,.015)
    axes[0].set_ylim(low-pad,high+pad)
    axes[0].set_xlim(0,measured.latency_p95_ms.max()*1.3)
    axes[0].set_title('Zoom: measured methods')
    axes[0].set_ylabel('Validation macro-F1 (%) — zoomed range')
    axes[1].axvline(100,color='red',linestyle='--',label='p95 budget 100 ms')
    axes[1].set_xlim(0,max(105,measured.latency_p95_ms.max()*1.1))
    axes[1].set_title('Same results against latency budget')
    axes[1].legend(loc='lower right')
    for ax in axes:
        ax.set_xlabel('Latency p95, batch 1 (ms)')
        ax.grid(alpha=.15)
    fig.suptitle('T04 ConvNeXt-Tiny | fold 0 | seed 0 | GPU pipeline, resident input')
    fig.tight_layout()
    output = Path(cfg.curves_dir)/'inference'/session_id/'f1_vs_latency_p95_detail.png'
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,bbox_inches='tight')
    plt.close(fig)
    return output
