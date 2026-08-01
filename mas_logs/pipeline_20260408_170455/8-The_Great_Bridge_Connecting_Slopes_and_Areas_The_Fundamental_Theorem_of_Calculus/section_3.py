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

class Section3Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        self.setup_layout(
            "Accumulation: Building the Area Function", 
            [
                'Imagine area growing as a vertical line moves.', 
                'We call this growing space the Area Function, A(x).', 
                'As x increases, more area is added.', 
                'A(x) tracks the total accumulated area from the start.', 
                'Watch how A(x) changes as the scrubber moves.'
            ]
        )

        # Mathematical parameters
        f_func = lambda x: 0.5 * x + 1
        a_func = lambda x: 0.25 * (x**2) + x
        x_min, x_max = 0, 4
        
        # 2. Preparation (Axes and Graph)
        axes_f = Axes(
            x_range=[0, 5, 1], y_range=[0, 4, 1], 
            x_length=4, y_length=2,
            axis_config={"include_tip": False, "font_size": 18}
        )
        axes_A = Axes(
            x_range=[0, 5, 1], y_range=[0, 10, 2], 
            x_length=4, y_length=2,
            axis_config={"include_tip": False, "font_size": 18}
        )
        
        # Place graph labels
        label_f = Text("f(x)", color="#00FF00", font_size=16).next_to(axes_f, RIGHT, buff=0.1)
        label_A = Text("A(x)", color="#FFEA00", font_size=16).next_to(axes_A, RIGHT, buff=0.1)
        
        axes_group = VGroup(axes_A, axes_f).arrange(DOWN, buff=0.8)
        # Issue 42: Place axes_group in area 'B2'-'E6'
        self.place_in_area(axes_group, 'B2', 'E6', scale_factor=1.0)
        
        f_graph = axes_f.plot(f_func, x_range=[0, 4], color="#00FF00")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.play(Create(axes_f), Create(f_graph), Write(label_f))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFEA00")
        self.play(Create(axes_A), Write(label_A))
        self.wait(1)

        # 3. Setup Accumulation Logic
        x_tracker = ValueTracker(0.1)
        
        # Scrubber [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/scrubber.svg]
        scrubber = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/scrubber.svg", color=WHITE)
        scrubber.scale(0.2)
        scrubber.move_to(axes_f.c2p(x_tracker.get_value(), 0))
        
        # Area shading
        area = always_redraw(lambda: axes_f.get_area(
            f_graph, x_range=[0, x_tracker.get_value()], color="#66BB6A", opacity=0.5
        ))
        
        # A(x) graph plotting
        a_graph = always_redraw(lambda: axes_A.plot(
            a_func, x_range=[0, x_tracker.get_value()], color="#FFEA00"
        ))
        
        # Updater for scrubber position
        scrubber.add_updater(lambda m: m.move_to(axes_f.c2p(x_tracker.get_value(), 0)))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#66BB6A")
        self.play(FadeIn(scrubber), FadeIn(area))
        self.play(x_tracker.animate.set_value(2), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFEA00")
        self.play(Create(a_graph))
        self.play(x_tracker.animate.set_value(4), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        
        # Indicator showing A'(x) = f(x)
        # Height of f(x) at x=4
        curr_x = 4
        vertical_line = Line(
            axes_f.c2p(curr_x, 0), axes_f.c2p(curr_x, f_func(curr_x)),
            color=WHITE, stroke_width=2
        )
        
        # Slope/Height Label
        # Issue 41: Place deriv_label at 'C5' with scale 0.7
        deriv_label = Text("Slope of A(x) = Height of f(x)", font_size=20, color=WHITE)
        self.place_at_grid(deriv_label, 'C5', scale_factor=0.7)
        
        self.play(Create(vertical_line), Write(deriv_label))
        self.wait(1)

        # Final Formula Box
        # Issue 40: Place ftc_box in area 'F2'-'F6' with scale 0.8
        ftc_formula = Text("A(x) = Integral of f(t) from 0 to x", font_size=24, color="#FFEA00")
        ftc_box = SurroundingRectangle(ftc_formula, color="#66BB6A", buff=0.1)
        ftc_group = VGroup(ftc_box, ftc_formula)
        self.place_in_area(ftc_group, 'F2', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(ftc_group))
        self.wait(3)

if __name__ == "__main__":
    pass
