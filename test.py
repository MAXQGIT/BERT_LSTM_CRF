sentence = '2011年7月26日，雷声公司研发的AGM-154联合防区外武器C-1，顺利通过了美国海军在慕古角海军靶场举行的首次自由飞行测试。试验中，JSOWC-1制导炸弹从一架F/A-18F超级大黄蜂战斗机上投放，飞行前段采用GPS/INS制导，末段转换为红外导引头制导，成功击中了太平洋上一艘79.2米长的移动无人舰艇。加装了Link16数据链的JSOWC-1，是美国现有武器装备中唯一具备网络能力、能够精确打击海上目标的武器。本次自由飞行测试结束之后，JSOWC-1还将接受一次综合性自由飞行测试和一次实战测试。按计划，具有发射后不管能力的JSOWC-1将于2013年具备初始作战能力。'

sentence_list = sentence.replace('。','。\n').split('\n')
print(sentence_list)


a=[]
c =[1,2]
a+=c
print(a)