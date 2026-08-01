from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        # LECTURE LINES
        lines = [
            "The triangle's shape naturally guides your mathematical thinking.",
            "Moving from Base to Exponent points to the Result.",
            "Position dictates the function, making symbols unnecessary."
        ]
        
        self.setup_layout("The Visual Logic of Operations", lines)
        
        # COLORS
        BASE_COLOR = "#3399FF" # Blue
        EXP_COLOR = "#FF8C00"  # Orange
        RES_COLOR = "#CC66FF"  # Purple
        SPARK_COLOR = "#FFFF00" # Yellow
        TRI_COLOR = GREY_B

        # GRID POSITIONS (Updated per Issue 32)
        p_base = self.grid["E2"]
        p_exp = self.grid["C4"]
        p_res = self.grid["E6"]

        # MOBJECTS
        triangle = Polygon(p_base, p_exp, p_res, color=TRI_COLOR, stroke_width=4)
        
        # Numbers
        base_num = MathTex("2", color=BASE_COLOR, font_size=48)
        exp_num = MathTex("3", color=EXP_COLOR, font_size=48)
        res_num = MathTex("8", color=RES_COLOR, font_size=48)
        
        self.place_at_grid(base_num, "E2")
        self.place_at_grid(exp_num, "C4") # Issue 32: exp_num to C4
        self.place_at_grid(res_num, "E6")
        
        # Labels
        base_label = Text("Base", color=BASE_COLOR, font_size=20)
        exp_label = Text("Exponent", color=EXP_COLOR, font_size=20)
        res_label = Text("Result", color=RES_COLOR, font_size=20)
        
        self.place_at_grid(base_label, "F2", scale_factor=0.8) # Issue 33
        self.place_at_grid(exp_label, "B4", scale_factor=0.8) # Issue 32
        self.place_at_grid(res_label, "F6", scale_factor=0.8) # Issue 33

        # Spark and Paths (Issue 24)
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/spark.svg]
        spark = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spark.svg")
        spark.set_color(SPARK_COLOR)
        spark.scale(0.2)
        
        # Spark halo for visibility
        spark_halo = Dot(color=SPARK_COLOR, radius=0.25, fill_opacity=0.3)
        spark_group = VGroup(spark_halo, spark)
        
        # Initial states
        res_num.set_opacity(0.3)
        res_label.set_opacity(0.3)

        # === Animation for Lecture Line 1 ===
        # The triangle's shape naturally guides your mathematical thinking.
        self.lecture[0].set_color(TRI_COLOR)
        self.play(
            Create(triangle),
            FadeIn(base_num), FadeIn(base_label),
            FadeIn(exp_num), FadeIn(exp_label),
            FadeIn(res_num), FadeIn(res_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Moving from Base to Exponent points to the Result.
        self.play(self.lecture[1].animate.set_color(SPARK_COLOR), run_time=0.5)
        
        # Spark travel Base -> Exponent
        spark_group.move_to(p_base)
        self.play(FadeIn(spark_group), run_time=0.5)
        self.play(spark_group.animate.move_to(p_exp), run_time=1.5, rate_func=smooth)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Position dictates the function, making symbols unnecessary.
        self.play(self.lecture[2].animate.set_color(RES_COLOR), run_time=0.5)
        
        # Spark travel Exponent -> Result
        self.play(spark_group.animate.move_to(p_res), run_time=1.5, rate_func=smooth)
        
        # Illuminate Result
        self.play(
            res_num.animate.set_opacity(1.0).scale(1.2),
            res_label.animate.set_opacity(1.0),
            Flash(p_res, color=RES_COLOR, line_length=0.4),
            FadeOut(spark_group),
            run_time=1
        )
        self.play(res_num.animate.scale(1/1.2), run_time=0.5)
        self.wait(3)
