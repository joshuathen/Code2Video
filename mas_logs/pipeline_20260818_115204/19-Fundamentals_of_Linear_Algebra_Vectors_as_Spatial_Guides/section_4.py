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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Basis vectors i and j span space.", "Combine them to reach any point.", "Every location is a linear combination."]
        self.setup_layout("Linear Combinations: The Basis of Space", lecture_lines)
        
        # Grid setup for visualization
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_numbers": False})
        # Use Asset as requested
        icon_grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # Applying layout improvements based on feedback
        self.place_in_area(grid, 'C2', 'F6', scale_factor=0.6)
        self.add(grid)
        self.place_at_grid(icon_grid, 'B2', scale_factor=0.3)
        self.add(icon_grid)

        i_vec = Vector(RIGHT, color=WHITE)
        j_vec = Vector(UP, color=WHITE)
        
        # Align with origin of the grid (0,0)
        origin = grid.c2p(0, 0)
        i_vec.shift(origin - i_vec.get_start())
        j_vec.shift(origin - j_vec.get_start())

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(i_vec), FadeIn(j_vec))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Scale i and j to form point (2, 1)
        i_scaled = Vector(2 * RIGHT, color="#FFFF00")
        j_scaled = Vector(1 * UP, color="#FFFF00")
        
        i_scaled.shift(origin - i_scaled.get_start())
        j_scaled.shift(origin + 2 * RIGHT - j_scaled.get_start())
        
        self.play(ReplacementTransform(i_vec.copy(), i_scaled), ReplacementTransform(j_vec.copy(), j_scaled))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        point = Dot(grid.c2p(2, 1), color="#FF00FF")
        self.play(Create(point))
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
