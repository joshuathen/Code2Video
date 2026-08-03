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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout("The Big Idea: Superposition", [
            "Superposition allows us to stack separate pure waves.",
            "Watch how three distinct frequencies align vertically.",
            "We add their heights together at every single point.",
            "The sum forms a new, more complex wave shape.",
            "More frequencies make the approximation even more accurate."
        ])
        
        # Define shared parameters
        x_min, x_max = -2, 2
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create three separate axes and waves
        # Colors: Wave 1 (#FF00FF), Wave 2 (#00FF00), Wave 3 (#0000FF)
        # Fix for Issue 29: Vertical spacing and column margin
        axes1 = Axes(x_range=[x_min, x_max], y_range=[-1.2, 1.2], x_length=4, y_length=0.8, axis_config={"include_tip": False, "stroke_width": 2})
        self.place_in_area(axes1, 'A2', 'A6')
        wave1 = axes1.plot(lambda x: 1.0 * np.sin(1 * x * PI), color="#FF00FF")
        
        axes2 = Axes(x_range=[x_min, x_max], y_range=[-0.5, 0.5], x_length=4, y_length=0.8, axis_config={"include_tip": False, "stroke_width": 2})
        self.place_in_area(axes2, 'C2', 'C6')
        wave2 = axes2.plot(lambda x: (1/3) * np.sin(3 * x * PI), color="#00FF00")
        
        axes3 = Axes(x_range=[x_min, x_max], y_range=[-0.3, 0.3], x_length=4, y_length=0.8, axis_config={"include_tip": False, "stroke_width": 2})
        self.place_in_area(axes3, 'E2', 'E6')
        wave3 = axes3.plot(lambda x: (1/5) * np.sin(5 * x * PI), color="#0000FF")
        
        self.play(Create(axes1), Create(wave1))
        self.play(Create(axes2), Create(wave2))
        self.play(Create(axes3), Create(wave3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Show vertical alignment indicators
        alignment_lines = VGroup(*[
            DashedLine(axes1.c2p(x, 1.2), axes3.c2p(x, -1.2), stroke_width=1, color=GRAY)
            for x in np.linspace(x_min, x_max, 7)
        ])
        self.play(Create(alignment_lines))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Show specific vertical height lines (summation at a point)
        sample_x = 0.5
        v_line1 = Line(axes1.c2p(sample_x, 0), axes1.c2p(sample_x, 1.0 * np.sin(1 * sample_x * PI)), color="#FF00FF", stroke_width=4)
        v_line2 = Line(axes2.c2p(sample_x, 0), axes2.c2p(sample_x, (1/3) * np.sin(3 * sample_x * PI)), color="#00FF00", stroke_width=4)
        v_line3 = Line(axes3.c2p(sample_x, 0), axes3.c2p(sample_x, (1/5) * np.sin(5 * sample_x * PI)), color="#0000FF", stroke_width=4)
        
        self.play(Create(v_line1), Create(v_line2), Create(v_line3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Create a single baseline axes for the sum
        # Fix for Issue 30: Positioning and scaling
        sum_axes = Axes(x_range=[x_min, x_max], y_range=[-1.5, 1.5], x_length=5, y_length=3.0, axis_config={"include_tip": False, "stroke_width": 2})
        self.place_in_area(sum_axes, 'B2', 'E6', scale_factor=0.8)
        
        # Define functions for transition
        f1 = lambda x: 1.0 * np.sin(1 * x * PI)
        f2 = lambda x: (1/3) * np.sin(3 * x * PI)
        f3 = lambda x: (1/5) * np.sin(5 * x * PI)
        f_total = lambda x: f1(x) + f2(x) + f3(x)

        # Transition waves to the central baseline
        collapsed_w1 = sum_axes.plot(f1, color="#FF00FF").set_opacity(0.3)
        collapsed_w2 = sum_axes.plot(f2, color="#00FF00").set_opacity(0.3)
        collapsed_w3 = sum_axes.plot(f3, color="#0000FF").set_opacity(0.3)
        resultant_wave = sum_axes.plot(f_total, color=WHITE)
        
        self.play(
            FadeOut(alignment_lines), FadeOut(v_line1), FadeOut(v_line2), FadeOut(v_line3),
            FadeOut(axes1), FadeOut(axes2), FadeOut(axes3),
            Transform(wave1, collapsed_w1),
            Transform(wave2, collapsed_w2),
            Transform(wave3, collapsed_w3),
            Create(sum_axes),
            run_time=2
        )
        
        self.play(Create(resultant_wave), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Square wave approximation (more frequencies)
        def square_approx(x, terms):
            return sum([(1/n) * np.sin(n * x * PI) for n in range(1, terms*2, 2)])
        
        higher_approx_wave = sum_axes.plot(lambda x: square_approx(x, 10), color=YELLOW)
        
        self.play(Transform(resultant_wave, higher_approx_wave), run_time=2)
        self.wait(2)
        
        # Finish
        self.lecture[4].set_color(WHITE)
        self.wait(2)
