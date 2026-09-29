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
            "We start with a simple square subdivision.",
            "Iteratively connect the midpoints of each sub-square.",
            "Repeating this creates an increasingly dense folding path."
        ]
        self.setup_layout("The Recursive Construction: The Peano/Hilbert Curve", lecture_lines)

        # Assets
        grid_bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        map_final = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        
        self.place_in_area(grid_bg, 'B4', 'E6', scale_factor=0.6)
        sq = Square(side_length=2, color="#FFFFFF")
        self.place_in_area(sq, 'B4', 'E6', scale_factor=0.6)
        
        self.play(FadeIn(grid_bg), Create(sq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00CED1")
        # Hilbert curve iteration 1
        path1 = VGroup(
            Line(np.array([-0.5, 0.5, 0]), np.array([-0.5, -0.5, 0])),
            Line(np.array([-0.5, -0.5, 0]), np.array([0.5, -0.5, 0])),
            Line(np.array([0.5, -0.5, 0]), np.array([0.5, 0.5, 0]))
        ).set_color("#00CED1")
        
        # Use constraint for line 2
        self.place_in_area(path1, 'A2', 'D5', scale_factor=0.75)
        self.play(Create(path1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        # Hilbert curve iteration 2 - simplified
        path2 = VGroup(
            Line(np.array([-0.75, 0.75, 0]), np.array([-0.75, 0.25, 0])),
            Line(np.array([-0.75, 0.25, 0]), np.array([-0.25, 0.25, 0])),
            Line(np.array([-0.25, 0.25, 0]), np.array([-0.25, 0.75, 0])),
            Line(np.array([-0.25, 0.75, 0]), np.array([0.25, 0.75, 0])),
            Line(np.array([0.25, 0.75, 0]), np.array([0.25, 0.25, 0])),
            Line(np.array([0.25, 0.25, 0]), np.array([0.75, 0.25, 0])),
            Line(np.array([0.75, 0.25, 0]), np.array([0.75, 0.75, 0]))
        ).set_color("#FF4500")
        
        self.place_in_area(path2, 'B2', 'E5', scale_factor=0.7)
        
        # Final transition to map asset
        self.play(Transform(path1, path2))
        self.play(FadeOut(sq), FadeOut(path1), FadeIn(map_final))
        self.place_in_area(map_final, 'B2', 'E5', scale_factor=0.7)
        self.wait(2)
