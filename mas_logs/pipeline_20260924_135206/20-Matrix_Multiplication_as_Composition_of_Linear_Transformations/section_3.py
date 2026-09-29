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
        self.setup_layout("Visualizing the Chain Reaction", [
            "Track the basis vectors i and j.",
            "B moves basis vectors to new positions.",
            "A then moves those results again.",
            "Final positions define the columns of matrix C.",
            "This chain is the matrix product AB."
        ])
        
        # Setup grid and basis vectors
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], x_length=4, y_length=4, axis_config={"include_tip": True})
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], x_length=4, y_length=4)
        
        # Assets
        vector_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vectors.svg")
        
        i_hat = Vector(RIGHT, color="#FF0000") # Asset color requirements
        j_hat = Vector(UP, color="#00FF00")
        
        grid_group = VGroup(grid, axes, i_hat, j_hat)
        self.place_in_area(grid_group, 'C2', 'F6', scale_factor=0.6)
        
        self.play(Create(grid), Create(axes), GrowArrow(i_hat), GrowArrow(j_hat))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        b_matrix = [[1.5, 0], [0, 0.7]]
        self.play(
            grid.animate.apply_matrix(b_matrix),
            i_hat.animate.apply_matrix(b_matrix),
            j_hat.animate.apply_matrix(b_matrix)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        a_matrix = [[0, -1], [1, 0]]
        self.play(
            grid.animate.apply_matrix(a_matrix),
            i_hat.animate.apply_matrix(a_matrix),
            j_hat.animate.apply_matrix(a_matrix)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        dot_i = Dot(i_hat.get_end(), color="#FFFF00")
        dot_j = Dot(j_hat.get_end(), color="#FFFF00")
        label_c = Text("Columns of C", font_size=24, color=WHITE)
        self.place_at_grid(label_c, 'E4', scale_factor=0.9)
        
        self.play(Create(dot_i), Create(dot_j), Write(label_c))
        
        # Place icon near label as per request
        icon_c = vector_icon.copy()
        self.place_at_grid(icon_c, 'E5', scale_factor=0.3)
        self.add(icon_c)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        self.play(Indicate(grid_group))
        self.wait(2)
