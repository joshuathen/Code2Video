from manim import *
import numpy as np

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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The 'Orange Peel' Effect",
            [
                "Imagine an orange in a high-dimensional space.",
                "Almost all volume concentrates near the surface boundary.",
                "Even a thin peel contains nearly all the mass.",
                "The interior of the sphere is essentially empty space.",
                "This is the counter-intuitive \"orange peel\" effect."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        outer_radius = 1.5
        inner_radius_init = 1.4
        
        fruit = Circle(radius=inner_radius_init, color="#FFA500", fill_opacity=0.4, stroke_width=0)
        peel = Annulus(inner_radius=inner_radius_init, outer_radius=outer_radius, color="#FFA500", fill_opacity=1.0, stroke_width=1)
        orange = VGroup(fruit, peel)
        
        self.place_in_area(orange, "B2", "E5")
        center_pt = orange.get_center()
        
        self.play(FadeIn(orange))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        line_r = Line(center_pt, center_pt + RIGHT * outer_radius, color=WHITE)
        label_r = MathTex("r", color=WHITE, font_size=24).next_to(line_r.get_end(), RIGHT, buff=0.1)
        
        line_inner = Line(center_pt, center_pt + RIGHT * inner_radius_init, color=WHITE).shift(DOWN * 0.1)
        label_inner = MathTex("r - \\epsilon", color=WHITE, font_size=24).next_to(line_inner.get_end(), DOWN, buff=0.1)
        
        labels = VGroup(line_r, label_r, line_inner, label_inner)
        
        self.play(Create(line_r), Write(label_r))
        self.play(Create(line_inner), Write(label_inner))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        formula = MathTex(
            "\\frac{V_{fruit}}{V_{total}} = (1 - \\epsilon)^n \\to 0",
            color="#FFFF00",
            font_size=32
        )
        self.place_at_grid(formula, "A4")
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        n_val = ValueTracker(3)
        # Using DecimalNumber for efficient updates instead of MathTex in always_redraw
        n_label_static = MathTex("n = ", color=WHITE, font_size=28).next_to(formula, DOWN)
        n_val_display = DecimalNumber(3, num_decimal_places=0, color=WHITE, font_size=28).next_to(n_label_static, RIGHT)
        n_val_display.add_updater(lambda d: d.set_value(n_val.get_value()))
        
        n_group = VGroup(n_label_static, n_val_display)
        
        def update_orange(mob):
            curr_n = n_val.get_value()
            start_r = 1.4
            end_r = 0.15 
            alpha = (curr_n - 3) / 97
            new_inner_r = start_r * (1 - alpha) + end_r * alpha
            
            mob[0].become(Circle(radius=new_inner_r, color="#FFA500", fill_opacity=0.4, stroke_width=0).move_to(center_pt))
            mob[1].become(Annulus(inner_radius=new_inner_r, outer_radius=outer_radius, color="#FFA500", fill_opacity=1.0, stroke_width=1).move_to(center_pt))

        orange.add_updater(update_orange)
        
        self.play(FadeOut(labels), FadeIn(n_group))
        self.play(n_val.animate.set_value(100), run_time=4, rate_func=linear)
        self.wait(1)
        
        orange.remove_updater(update_orange)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        final_peel = Annulus(
            inner_radius=orange[1].inner_radius, 
            outer_radius=outer_radius, 
            color="#FFD700", 
            fill_opacity=1.0, 
            stroke_width=2
        ).move_to(center_pt)
        
        self.play(
            FadeOut(orange[0]),
            Transform(orange[1], final_peel),
            FadeOut(n_group)
        )
        
        self.play(orange[1].animate.set_stroke(width=4), run_time=1)
        self.wait(2)
