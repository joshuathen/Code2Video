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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "PDFs measure probability density, not probability.",
            "Think of density as scattered gold dust.",
            "Probability is the area under the curve."
        ]
        self.setup_layout("Defining the PDF", lecture_lines)
        
        # Assets
        gold_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gold.svg")
        dust_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dust.svg")
        
        # Axis setup
        axes = Axes(
            x_range=[0, 6, 1], y_range=[0, 4, 1],
            axis_config={"include_numbers": False, "tip_shape": StealthTip}
        ).scale(0.6)
        y_label = Text("Density", font_size=20).next_to(axes.y_axis.get_top(), RIGHT)
        x_label = Text("Value", font_size=20).next_to(axes.x_axis.get_right(), UP)
        
        plot_group = VGroup(axes, y_label, x_label)
        self.place_in_area(plot_group, "B2", "E5", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        gold_icon.scale(0.5).to_edge(UP, buff=0.5).shift(RIGHT * 2)
        self.play(FadeIn(gold_icon))
        self.play(Create(axes), Write(y_label), Write(x_label))
        func = lambda x: 3 * np.exp(-(x - 3)**2 / 0.5)
        curve = axes.plot(func, x_range=[1, 5], color="#00CED1")
        self.play(Create(curve))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        
        # Highlight point x
        x_val = 3
        point_x = Dot(axes.c2p(x_val, 0), color=RED)
        label_x = MathTex("x", font_size=24).next_to(point_x, DOWN)
        
        f_x_point = Dot(axes.c2p(x_val, func(x_val)), color=RED)
        label_fx = MathTex("f(x)", font_size=24).next_to(f_x_point, UP)
        
        dashed_line = DashedLine(point_x.get_center(), f_x_point.get_center(), color=WHITE)
        dust_icon.scale(0.3).next_to(dashed_line, LEFT)
        
        self.play(FadeIn(point_x), Write(label_x))
        self.play(Create(dashed_line), FadeIn(dust_icon), FadeIn(f_x_point), Write(label_fx))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Highlight area
        area = axes.get_area(curve, x_range=[2.5, 3.5], color=YELLOW, opacity=0.3)
        self.play(Create(area))
        self.wait(2)
