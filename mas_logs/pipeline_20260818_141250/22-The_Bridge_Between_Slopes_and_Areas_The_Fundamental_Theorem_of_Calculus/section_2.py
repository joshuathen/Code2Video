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
        self.setup_layout("The Inverse Relationship: Integration vs. Differentiation", 
                          ["Integration and differentiation are inverses.", 
                           "Much like addition and subtraction.", 
                           "Define F(x) as accumulation."])
        
        # Assets
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        scale = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        axes = Axes(x_range=[-2, 2], y_range=[-1, 3], axis_config={"include_tip": False})
        f = axes.plot(lambda x: x**2 + 0.5, color=BLUE)
        df = axes.plot(lambda x: 2*x, color=RED)
        
        f_label = MathTex("f(x)", color=BLUE).scale(0.8)
        df_label = MathTex("f'(x)", color=RED).scale(0.8)
        
        self.place_at_grid(f, "B3", scale_factor=0.6)
        self.place_at_grid(f_label, "B2")
        self.place_at_grid(df, "E3", scale_factor=0.6)
        self.place_at_grid(df_label, "E2")
        self.place_at_grid(calculator, "A6", scale_factor=0.3)
        
        self.play(Create(f), Write(f_label), Create(df), Write(df_label), FadeIn(calculator))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.place_at_grid(ruler, "D6", scale_factor=0.3)
        self.play(FadeIn(ruler))
        
        tracker = ValueTracker(-1.5)
        
        # Persistent mobject for tangent
        tangent = Line(start=LEFT, end=RIGHT, color=YELLOW).scale(0.3)
        self.add(tangent)
        
        def update_tangent(m):
            x = tracker.get_value()
            slope = 2 * x
            point = axes.c2p(x, x**2 + 0.5)
            m.set_angle(np.arctan(slope))
            m.move_to(point)
            
        tangent.add_updater(update_tangent)
        self.play(tracker.animate.set_value(1.5), run_time=3)
        tangent.remove_updater(update_tangent)
        self.remove(tangent, ruler)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.place_at_grid(scale, "C6", scale_factor=0.3)
        self.play(FadeIn(scale))
        
        area = axes.get_area(f, x_range=[-1, 1], color=GREEN, opacity=0.3)
        f_label_int = MathTex(r"F(x) = \int_a^x f(t) dt", color=GREEN).scale(0.7)
        
        area_group = VGroup(area, f_label_int)
        self.place_in_area(area_group, "B4", "E6", scale_factor=0.8)
        
        self.play(FadeIn(area), Write(f_label_int))
        self.wait(2)
