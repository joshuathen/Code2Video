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
        lecture_lines = [
            "Left side displays the original function curve.",
            "Right side traces the derivative function path.",
            "This illustrates the derivative as a machine."
        ]
        self.setup_layout("Visualizing the Mapping (The Derivative Machine)", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        axes1 = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False}).scale(0.5)
        axes2 = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False}).scale(0.5)
        
        self.place_at_grid(axes1, 'B2', scale_factor=0.6)
        self.place_at_grid(axes2, 'B5', scale_factor=0.6)
        
        input_label = Text("Input", font_size=20).next_to(axes1, UP, buff=0.1)
        deriv_label = Text("Derivative", font_size=20).next_to(axes2, UP, buff=0.1)
        
        machine_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg").scale(0.5)
        self.place_at_grid(machine_icon, 'E4', scale_factor=0.3)
        
        self.play(FadeIn(axes1), FadeIn(axes2), Write(input_label), Write(deriv_label), FadeIn(machine_icon))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        curve = axes1.plot(lambda x: 0.5 * x**2, color=BLUE)
        point = Dot(axes1.c2p(-1.5, 0.5 * (-1.5)**2), color=YELLOW)
        tracker = ValueTracker(-1.5)
        
        point.add_updater(lambda m: m.move_to(axes1.c2p(tracker.get_value(), 0.5 * tracker.get_value()**2)))
        
        deriv_curve = axes2.plot(lambda x: x, color=ORANGE)
        deriv_trace = TracedPath(point.get_center, stroke_color=ORANGE) # Simplified logic
        
        self.play(Create(curve))
        self.play(tracker.animate.set_value(1.5), run_time=3)
        self.lecture[1].set_color(ORANGE)

        # === Animation for Lecture Line 3 ===
        self.play(Create(deriv_curve))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
