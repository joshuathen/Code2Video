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
        lecture_lines = ["Linear combinations scale and add vectors.", "Span is every reachable destination.", "Varied factors create a grid."]
        self.setup_layout("Linear Combinations and Span", lecture_lines)
        
        # Background Asset
        grid_bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_bg, 'A2', 'E5', scale_factor=0.85)
        self.add(grid_bg)

        # Define vectors
        u = Vector(RIGHT * 1.5 + UP * 0.5, color="#33FF57")
        v = Vector(RIGHT * 0.5 + UP * 1.5, color="#3357FF")
        vector_group = VGroup(u, v)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#33FF57")
        self.place_in_area(vector_group, 'D1', 'F3', scale_factor=0.6)
        self.play(Create(u), Create(v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        u_end = u.get_end()
        v_end = v.get_end()
        w_end = u_end + v_end
        
        line1 = DashedLine(u_end, w_end, color="#FFFFFF")
        line2 = DashedLine(v_end, w_end, color="#FFFFFF")
        self.play(Create(line1), Create(line2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF33")
        w = Vector(w_end, color="#FFFF33")
        self.play(Create(w))
        self.wait(2)
