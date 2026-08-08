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
        # Colors based on storyboard
        color_base = "#3498DB"      # Blue
        color_exponent = "#E74C3C"  # Red
        color_result = "#2ECC71"    # Green
        
        self.setup_layout("Prerequisite: The Three Roles", [
            "Every exponential relationship involves three specific roles.",
            "The Base is our starting multiplier or root.",
            "The Exponent counts growth, and Result is the value."
        ])
        
        # === Animation for Lecture Line 1 ===
        # "Every exponential relationship involves three specific roles."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        # "The Base is our starting multiplier or root."
        # Storyboard: Show "Base" in #3498DB at bottom-left.
        # Layout fix: Place at D2 using place_at_grid per Issue 40.
        base_label = Text("Base", color=color_base, font_size=36)
        self.place_at_grid(base_label, 'D2', scale_factor=1.0)
        
        # Highlight "Base" in lecture line 2: "The Base is our starting multiplier or root."
        # "Base" is index 4:8
        self.play(
            self.lecture[1][4:8].animate.set_color(color_base),
            Write(base_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The Exponent counts growth, and Result is the value."
        # Storyboard: Show "Exponent" in #E74C3C at the top.
        # Storyboard: Show "Result" in #2ECC71 at bottom-right.
        # Layout fix: Exponent at B3, Result at D5 per Issue 40.
        
        exponent_label = Text("Exponent", color=color_exponent, font_size=36)
        result_label = Text("Result", color=color_result, font_size=36)
        
        self.place_at_grid(exponent_label, 'B3', scale_factor=1.0)
        self.place_at_grid(result_label, 'D5', scale_factor=1.0)
        
        # Highlight "Exponent" and "Result" in lecture line 3:
        # "The Exponent counts growth, and Result is the value."
        # "Exponent" index 4:12, "Result" index 32:38
        self.play(
            self.lecture[2][4:12].animate.set_color(color_exponent),
            self.lecture[2][32:38].animate.set_color(color_result),
            Write(exponent_label),
            Write(result_label)
        )
        self.wait(3)
