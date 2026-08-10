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
        self.setup_layout("Prerequisite Review: The 'Language' of Coordinates", 
                          ["A basis is just our coordinate grid.", 
                           "Vectors are arrows, independent of grids.", 
                           "Different grids give different coordinate descriptions."])
        
        # Grid setup
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg
        # Using SVG for grid as per storyboard
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        grid_asset.set_color(WHITE)
        
        i_hat = Vector(RIGHT, color=YELLOW)
        j_hat = Vector(UP, color=BLUE)
        basis = VGroup(grid_asset, i_hat, j_hat)
        
        v = Vector([1.5, 1], color=RED)
        vector_labels = VGroup(MathTex("x", color=YELLOW), MathTex("y", color=BLUE)).arrange(RIGHT)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(basis, 'C3', 'F6', scale_factor=0.6)
        self.play(Create(grid_asset), GrowArrow(i_hat), GrowArrow(j_hat))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(v, 'C3', scale_factor=0.7)
        self.play(GrowArrow(v))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RED)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(vector_labels, 'C4', scale_factor=0.5)
        self.play(FadeIn(vector_labels))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        self.wait(2)
