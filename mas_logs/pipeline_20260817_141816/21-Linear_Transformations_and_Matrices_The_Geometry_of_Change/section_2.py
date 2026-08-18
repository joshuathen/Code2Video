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
            "Any transformation is defined by basis vectors.",
            "Observe the unit vectors i-hat and j-hat.",
            "Where they land determines the entire transformation."
        ]
        self.setup_layout("Defining the Basis Vectors", lecture_lines)
        
        # Setup Grid
        plane = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, 'C2', 'F5', scale_factor=0.45)
        
        i_hat = Vector(RIGHT, color=YELLOW)
        j_hat = Vector(UP, color=RED)
        i_label = MathTex(r"\hat{i}", color=YELLOW)
        j_label = MathTex(r"\hat{j}", color=RED)
        
        # Fix labels: use place_at_grid as requested
        self.place_at_grid(i_label, 'D3', scale_factor=0.7)
        self.place_at_grid(j_label, 'C4', scale_factor=0.7)
        
        # Position vectors relative to the plane center
        i_hat.move_to(plane.c2p(1,0))
        j_hat.move_to(plane.c2p(0,1))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(plane), GrowArrow(i_hat), GrowArrow(j_hat), Write(i_label), Write(j_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        self.play(Indicate(i_hat), Indicate(j_hat))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        # New basis vectors coordinates in the plane
        i_new_pos = plane.c2p(2, -1)
        j_new_pos = plane.c2p(0, 2)
        
        i_new = Vector(i_new_pos - plane.get_center(), color=YELLOW).move_to(i_new_pos)
        j_new = Vector(j_new_pos - plane.get_center(), color=RED).move_to(j_new_pos)
        
        self.play(
            ReplacementTransform(i_hat, i_new),
            ReplacementTransform(j_hat, j_new),
            i_label.animate.next_to(i_new, DOWN),
            j_label.animate.next_to(j_new, LEFT)
        )
        self.wait(2)
