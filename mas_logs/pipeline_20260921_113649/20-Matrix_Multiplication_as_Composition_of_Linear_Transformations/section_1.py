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
        self.setup_layout("Prerequisite Warm-up: The Geometric View of a Matrix", 
                          ["Matrices are functions that transform vectors.", 
                           "Imagine a plane transforming under a matrix.", 
                           "The unit square moves to a parallelogram."])
        
        # Define objects
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg").scale(0.6)
        self.place_in_area(plane, 'B2', 'D5', scale_factor=0.6)
        
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg").scale(0.5)
        self.place_at_grid(grid_asset, 'E3')
        
        i_hat = Vector([1, 0], color="#FF0000")
        j_hat = Vector([0, 1], color="#00FF00")
        
        # Start vectors
        self.add(plane, grid_asset)
        self.add(self.place_at_grid(i_hat, 'C4', scale_factor=0.6), self.place_at_grid(j_hat, 'C5', scale_factor=0.6))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#0000FF"))
        rect = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        self.place_in_area(rect, 'B2', 'D5', scale_factor=0.6)
        self.play(Create(rect))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # New vectors
        i_hat_new = Vector([1, 1], color="#FFFF00")
        j_hat_new = Vector([-1, 1], color="#FFFF00")
        
        self.place_at_grid(i_hat_new, 'D4', scale_factor=0.6)
        self.place_at_grid(j_hat_new, 'C5', scale_factor=0.6)
        
        self.play(Transform(i_hat, i_hat_new), Transform(j_hat, j_hat_new))
        self.wait(2)
