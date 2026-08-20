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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Data is organized in a grid.", "We scan with a small window.", "The window looks at local neighborhoods."]
        self.setup_layout("Prerequisites: The Sliding Window Concept", lecture_lines)
        
        # Create 1D array
        array_size = 8
        cells = VGroup(*[Square(side_length=0.6, color=WHITE) for _ in range(array_size)])
        cells.arrange(RIGHT, buff=0)
        # Apply fix from issue 21/36
        self.place_in_area(cells, 'D3', 'E5', scale_factor=0.7)
        
        # Create sliding window using SVG asset
        window = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/window.svg")
        window.set_color(YELLOW)
        window.scale(0.8) # Adjust scale to match cells
        window.move_to(cells[0])
        
        # Highlight rectangle
        highlight = SurroundingRectangle(cells[0], color="#FF4500", buff=0)
        
        # Initial State
        self.play(FadeIn(cells))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#87CEEB")) # SkyBlue
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#90EE90")) # LightGreen
        
        self.play(FadeIn(window), Create(highlight))
        
        # Slide the window
        for i in range(array_size - 1):
            target_pos = cells[i+1].get_center()
            target_rect = SurroundingRectangle(cells[i+1], color="#FF4500", buff=0)
            self.play(
                window.animate.move_to(target_pos),
                Transform(highlight, target_rect),
                run_time=0.5
            )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700")) # Gold
        
        # Highlight local neighborhood
        self.play(highlight.animate.set_stroke(width=6), run_time=1)
        self.wait(1)
        
        self.play(FadeOut(window), FadeOut(highlight))
        self.wait(2)
