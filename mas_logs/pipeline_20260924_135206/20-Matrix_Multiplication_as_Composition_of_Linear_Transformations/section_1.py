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
        self.setup_layout("Prerequisite Warm-up: The Transformation Perspective", 
                          ["Matrices act as functions moving the entire coordinate plane.", 
                           "A matrix A transforms a vector v into w.", 
                           "Think of this as a transformation of the grid."])
        
        # Load asset
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # Define objects
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], x_length=4, y_length=4)
        v_vec = Vector([1, 1], color=YELLOW)
        
        # Matrix to transform
        matrix = [[1, 1], [0, 1]] # Shear matrix
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(grid, 'B2', 'F6', scale_factor=0.75)
        self.play(FadeIn(grid_asset.move_to(grid.get_center())), Create(grid), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(v_vec, 'D3', scale_factor=0.7)
        v_label = self.add_labeled_vector(v_vec, "v", position='top_right', offset=0.2)
        self.play(Create(v_vec), Write(v_label))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        
        # Animate the grid transformation
        target_grid = grid.copy().apply_matrix(matrix)
        target_v = v_vec.copy().apply_matrix(matrix)
        
        self.play(
            Transform(grid, target_grid),
            Transform(v_vec, target_v),
            run_time=3
        )
        self.wait(1)
