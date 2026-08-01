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

class Section6Scene(TeachingScene):
    def construct(self):
        # Fetching data based on shared state
        title = "Real-World Application: Force Fields"
        lecture_lines = [
            "In reality, vectors represent forces like wind or gravity.",
            "A boat's path depends on the sum of all forces.",
            "Mastering vectors unlocks the secrets of the physical world."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors - using hex strings as per L008
        COLOR_ENGINE = "#00FFFF"
        COLOR_CURRENT = "#FF00FF"
        COLOR_RESULTANT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(COLOR_ENGINE))
        self.wait(2)

        # Represent a boat's engine force with a horizontal cyan vector
        # Engine force: Horizontal from C2 towards C4
        start_point = self.grid["C2"]
        end_point = self.grid["C4"]
        engine_vector = Arrow(start_point, end_point, buff=0, color=COLOR_ENGINE, stroke_width=6)
        
        engine_label = Text("Engine Force", font_size=20, color=COLOR_ENGINE)
        # Fix for Issue 41: Use place_in_area for multi-word label
        self.place_in_area(engine_label, "B3", "B4", scale_factor=0.8)

        self.play(Create(engine_vector))
        self.play(Write(engine_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(self.lecture[1].animate.set_color(COLOR_CURRENT))
        self.wait(2)

        # Add a vertical magenta vector representing the river's current force
        # Current force: Vertical from the tip of engine_vector (C4) towards E4
        curr_start = self.grid["C4"]
        curr_end = self.grid["E4"]
        current_vector = Arrow(curr_start, curr_end, buff=0, color=COLOR_CURRENT, stroke_width=6)
        
        current_label = Text("Current Force", font_size=20, color=COLOR_CURRENT)
        # Fix for Issue 42: Use place_in_area and move right to avoid overlap
        self.place_in_area(current_label, "D6", "E6", scale_factor=0.8)

        self.play(Create(current_vector))
        self.play(Write(current_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(self.lecture[2].animate.set_color(COLOR_RESULTANT))
        self.wait(2)

        # Draw the white resultant vector showing the boat's true diagonal path
        # Resultant: Diagonal from C2 to E4
        res_start = self.grid["C2"]
        res_end = self.grid["E4"]
        resultant_vector = Arrow(res_start, res_end, buff=0, color=COLOR_RESULTANT, stroke_width=8)
        
        resultant_label = Text("Resultant Path", font_size=20, color=COLOR_RESULTANT)
        # Fix for Issue 43: Move to E3-E4 to avoid left-side crowding and balance the composition
        self.place_in_area(resultant_label, "E3", "E4", scale_factor=0.8)

        self.play(Create(resultant_vector))
        self.play(Write(resultant_label))
        self.wait(2)
