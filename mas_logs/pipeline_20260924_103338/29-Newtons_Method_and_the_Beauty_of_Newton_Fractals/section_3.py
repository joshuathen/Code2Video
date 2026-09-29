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
            "Complex numbers allow functions to have multiple roots.",
            "Starting points determine which root the method hits.",
            "Regions form clear 'Basins of Attraction'."
        ]
        self.setup_layout("The Chaos of Multi-Root Systems", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        roots = VGroup(
            Dot(color="#FF0000"),
            Dot(color="#FF0000"),
            Dot(color="#FF0000")
        )
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg")
        self.place_at_grid(roots[0], 'B3', scale_factor=0.7)
        self.place_at_grid(roots[1], 'B5', scale_factor=0.7)
        self.place_at_grid(roots[2], 'C5', scale_factor=0.7)
        self.play(FadeIn(roots))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.place_at_grid(drone, 'F1', scale_factor=0.5)
        self.play(FadeIn(drone))
        path = CurvedArrow(self.grid['F1'], self.grid['B3'], angle=-TAU/4, color=WHITE)
        self.play(Create(path))
        self.play(FadeOut(path))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#800080")
        # Creating a placeholder for the region graphic
        region = Rectangle(width=2.5, height=2, color="#800080", fill_opacity=0.3)
        self.place_in_area(region, 'D3', 'F6', scale_factor=0.6)
        self.play(FadeIn(region))
        self.play(Indicate(region, color="#800080"))
        self.wait(1)
