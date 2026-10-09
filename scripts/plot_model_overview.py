"""Report illustration: causal features and graph in transaction anomaly detection."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/contextual-v4/seed-20261002/figures')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(14, 8.2))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 14), ylim=(0, 8.2))
    ax.axis('off')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=0.14',
                                   facecolor=color,edgecolor='#94a3b8',linewidth=1.2))
        ax.text(x+w/2,y+h-.28,title,ha='center',va='top',fontsize=12,fontweight='bold',color='#172554')
        ax.text(x+w/2,y+h/2-.12,body,ha='center',va='center',fontsize=10.5,linespacing=1.6,color='#1e293b')
    def arrow(start,end,label=None):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,color='#475569',linewidth=1.6))
        if label:
            ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.14,label,ha='center',fontsize=9,color='#475569')
    ax.text(7,7.9,'Vai trò của feature và graph trong phát hiện giao dịch bất thường',
            ha='center',fontsize=17,fontweight='bold',color='#0f172a')
    ax.text(7,7.5,'Thiết kế mô hình GATv2: học quan hệ tài khoản và ngữ cảnh của giao dịch',
            ha='center',fontsize=11,color='#475569')
    box(.3,3.15,2.5,2.2,'Dữ liệu quan sát','Giao dịch đang xét\n+\nLịch sử đã xảy ra\ntrước thời điểm chấm','#f1f5f9')
    box(3.65,4.65,3.7,2.1,'Feature giao dịch','Amount, giờ, nghiệp vụ, kênh\nTần suất giao dịch gần đây\nMedian amount theo nghiệp vụ\nNgười nhận mới, cold start','#eff6ff')
    box(3.65,1.8,3.7,2.1,'Graph lịch sử','Node: account / merchant / cash\nEdge: quan hệ dòng tiền\nChỉ lấy lịch sử trước giao dịch\nNode features: loại + lịch sử','#ecfdf5')
    box(8.1,1.8,2.4,2.1,'GATv2 encoder','Tổng hợp thông tin\ntừ các node lân cận\n↓\nBiểu diễn source / destination','#ecfdf5')
    box(8.1,4.65,2.4,2.1,'Kết hợp thông tin','Feature giao dịch\n+\nBiểu diễn hai node\n→ bộ phân loại MLP','#f5f3ff')
    box(11.35,4.65,2.3,2.1,'Kết quả dự đoán','Điểm anomaly\n↓ ngưỡng quyết định\nBình thường / bất thường','#fff7ed')
    arrow((2.92,4.6),(3.5,5.45))
    arrow((2.92,3.8),(3.5,2.9))
    arrow((7.48,5.7),(7.95,5.7))
    arrow((7.48,2.85),(7.95,2.85))
    arrow((9.3,4.02),(9.3,4.5))
    arrow((10.63,5.7),(11.2,5.7))
    ax.text(12.5,3.8,'Nhãn thật dùng riêng\nđể huấn luyện / đánh giá',ha='center',fontsize=10,color='#64748b',linespacing=1.5)
    box(.3,.2,13.35,1.05,'Ví dụ: cùng amount 5 triệu nhưng ngữ cảnh khác nhau',
        'Thường chuyển 5 triệu, ít giao dịch → có thể bình thường   |   Thường chuyển 200 nghìn, 6 lần / 2 phút, người nhận mới → đáng nghi', '#fffbeb')
    ax.text(7,1.5,'Nguyên tắc: không đưa nhãn, profile sinh dữ liệu hoặc kết quả hiện tại vào input.',
            ha='center',fontsize=10,color='#475569')
    args.output.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(pad=.7)
    for ext in ('png','svg','pdf'):
        fig.savefig(args.output/f'feature_graph_gatv2.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(args.output/'feature_graph_gatv2.png')


if __name__=='__main__':
    main()
