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
        # Colors
        color_input = "#6495ED"  # Cornflower Blue
        color_output = "#FFD700" # Gold
        color_shadow = "#FFA500" # Orange
        color_slow = "#FF4500"   # Orange Red

        self.setup_layout("Character Application: The Speed-Changing Chameleon", [
            "A chameleon walks steadily along the input line.",
            "His shadow moves faster where the scaling factor is high.",
            "The shadow crawls slowly in compressed regions."
        ])
        
        # Asset path
        chameleon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/chameleon.png"

        # Grid positions for lines
        # Input line on Row C, Output line on Row E
        line_start_x = self.grid["C1"]
        line_end_x = self.grid["C6"]
        line_start_y = self.grid["E1"]
        line_end_y = self.grid["E6"]

        # Background Lines
        input_line = Line(line_start_x, line_end_x, color=GREY_E)
        output_line = Line(line_start_y, line_end_y, color=GREY_E)
        
        # Labels
        input_label = Text("Input Space", font_size=20, color=color_input)
        # Resolved Issue 32: Centering input_label
        self.place_in_area(input_label, 'B2', 'B5', scale_factor=0.8)
        
        output_label = Text("Output Space", font_size=20, color=color_output)
        # Resolved Issue 33: Centering output_label
        self.place_in_area(output_label, 'D2', 'D5', scale_factor=0.8)

        # Value Tracker for the chameleon's position (x from -1 to 1)
        x_tracker = ValueTracker(-1)

        # Function f(x) = x^3. Derivative f'(x) = 3x^2.
        # High scaling at x = +/-1, Low scaling at x = 0.
        def f(x):
            return x**3

        # Mappings from internal x [-1, 1] to screen coordinates
        def get_input_pos(x):
            alpha = (x + 1) / 2
            return line_start_x + alpha * (line_end_x - line_start_x)

        def get_output_pos(x):
            y = f(x)
            alpha = (y + 1) / 2
            return line_start_y + alpha * (line_end_y - line_start_y)

        # Mobjects
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/chameleon.png]
        chameleon = ImageMobject(chameleon_path).scale(0.3)
        chameleon.add_updater(lambda m: m.move_to(get_input_pos(x_tracker.get_value())))

        # Shadow also uses [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/chameleon.png]
        shadow = ImageMobject(chameleon_path).scale(0.3).set_opacity(0.5)
        shadow.add_updater(lambda m: m.move_to(get_output_pos(x_tracker.get_value())))

        connector = Line(color=WHITE, stroke_opacity=0.2)
        connector.add_updater(lambda m: m.put_start_and_end_on(chameleon.get_center(), shadow.get_center()))

        # Initial State
        self.add(input_line, output_line, input_label, output_label)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_input)
        self.add(chameleon)
        # Chameleon walks steadily
        self.play(x_tracker.animate.set_value(1), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_output)
        self.add(shadow, connector)
        
        # Reset and show high scaling region (x from -1 to -0.5)
        x_tracker.set_value(-1)
        self.play(x_tracker.animate.set_value(-0.5), run_time=1.5, rate_func=linear)
        self.wait(0.5)
        
        # Jump ahead to the other high scaling region (x from 0.5 to 1)
        x_tracker.set_value(0.5)
        self.play(x_tracker.animate.set_value(1), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_slow)
        
        # Show low scaling region (x from -0.5 to 0.5)
        # Note: Same x-interval length (1.0), but shadow moves much less.
        x_tracker.set_value(-0.5)
        self.play(x_tracker.animate.set_value(0.5), run_time=4, rate_func=linear)
        self.wait(2)
