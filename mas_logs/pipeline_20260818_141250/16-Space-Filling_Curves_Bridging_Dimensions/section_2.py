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
        lecture_lines = [
            "Start with a simple grid of squares.",
            "Trace a path through each subdivision.",
            "Recursion creates infinite complexity."
        ]
        self.setup_layout("The Peano Curve Construction", lecture_lines)
        
        # Initial state setup
        grid = VGroup(*[Square(side_length=0.8, color=BLUE_D) for _ in range(9)]).arrange_in_grid(3, 3)
        self.place_in_area(grid, 'C3', 'E5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(grid), self.lecture[0].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Simple trace path
        path = VMobject(color=YELLOW)
        path.set_points_smoothly([grid[0].get_center(), grid[1].get_center(), grid[2].get_center(),
                                  grid[5].get_center(), grid[4].get_center(), grid[3].get_center(),
                                  grid[6].get_center(), grid[7].get_center(), grid[8].get_center()])
        self.play(Create(path), self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visualize recursion concept: sub-dividing
        new_grid = VGroup(*[Square(side_length=0.25, color=BLUE_E) for _ in range(81)]).arrange_in_grid(9, 9)
        self.place_in_area(new_grid, 'C3', 'E5', scale_factor=0.9)
        self.play(FadeOut(grid), FadeOut(path), FadeIn(new_grid), self.lecture[2].animate.set_color(RED))
        self.wait(2)
