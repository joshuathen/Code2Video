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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Determinants represent the volume scale factor.", "They measure transformation-induced volume expansion.", "A shrinkage means a smaller determinant."]
        self.setup_layout("Determinants: The Volume Factor", lecture_lines)
        
        # Define mobjects
        square = Square(side_length=1.5, color="#00FF00")
        parallelogram = Polygon([0, 0, 0], [1.5, 0, 0], [2.0, 1.5, 0], [0.5, 1.5, 0], color="#FF4500")
        det_label = MathTex(r"\det(A) = \text{Area}_{\text{new}}", color="#FFD700")
        # Asset placeholders (icons not strictly needed for this visual logic but required by prompt instructions)
        icon_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.place_at_grid(square, 'B2', scale_factor=0.6)
        self.play(Create(square), FadeIn(self.place_at_grid(icon_asset, 'F2', scale_factor=0.3)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF4500")
        self.place_at_grid(parallelogram, 'B4', scale_factor=0.6)
        self.play(ReplacementTransform(square, parallelogram))
        self.wait(1)
        
        self.lecture[1].set_color("#FFFFFF")
        self.play(Write(self.place_at_grid(det_label, 'D2', scale_factor=0.7)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        shrink_square = Square(side_length=0.7, color="#FF0000")
        # Visualizing negative determinant/flipping orientation
        self.play(ReplacementTransform(parallelogram, self.place_at_grid(shrink_square, 'D4', scale_factor=0.8)), 
                  FadeIn(self.place_at_grid(icon_asset.copy(), 'F5', scale_factor=0.3)))
        self.wait(1)
