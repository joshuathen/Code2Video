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
            "Wordle strategies aim to maximize information gain per guess.",
            "Each guess acts as a filter on candidate words.",
            "Optimal guesses minimize remaining entropy in the list."
        ]
        self.setup_layout("Applying Entropy to Wordle", lecture_lines)
        
        # Grid setup
        grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid, 'A3', 'C6', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.play(FadeIn(grid))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#808080")
        tile_gray = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tile.svg")
        tile_gray.set_color("#808080")
        # Placing tile relative to grid center or appropriate grid spot
        self.place_at_grid(tile_gray, 'B4')
        self.play(FadeIn(tile_gray))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        entropy_text = Text("Entropy Decreasing", font_size=24, color="#FFD700")
        self.place_at_grid(entropy_text, 'C3', scale_factor=0.7)
        arrow = Arrow(start=self.grid['D3'], end=self.grid['C4'], color="#FFD700")
        self.play(Write(entropy_text), GrowArrow(arrow))
        self.wait(2)
