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
            "Adding vectors combines their movements.",
            "Place the second tail at the first head.",
            "Form a parallelogram with these two vectors.",
            "The diagonal is the sum of vectors.",
            "Like a boat moving across a current."
        ]
        self.setup_layout("Vector Addition: The Parallelogram Rule", lecture_lines)
        
        # Assets
        boat = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/boat.svg")
        
        # Define vectors
        vec_a = Arrow(ORIGIN, RIGHT * 2 + UP * 1, color="#FFD700", buff=0)
        vec_b = Arrow(ORIGIN, RIGHT * 1 + UP * 2, color="#00CED1", buff=0)
        
        label_a = MathTex("A", color="#FFD700")
        label_b = MathTex("B", color="#00CED1")
        
        group_ab = VGroup(vec_a, vec_b, label_a, label_b, boat)
        
        # Apply critic fix: scale down group
        self.place_in_area(group_ab, 'B3', 'E5', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(Create(vec_a), Write(label_a), FadeIn(boat))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00CED1")
        vec_b_shifted = Arrow(vec_a.get_end(), vec_a.get_end() + vec_b.get_vector(), color="#00CED1", buff=0)
        self.play(Create(vec_b_shifted), Write(label_b))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        path = [vec_a.get_start(), vec_a.get_end(), vec_a.get_end() + vec_b.get_vector(), vec_b.get_vector() + vec_a.get_vector(), vec_a.get_start()]
        parallelogram = Polygon(*[vec_a.get_start(), vec_a.get_end(), vec_a.get_end() + vec_b.get_vector(), vec_b.get_vector()], color="#FFFFFF", stroke_opacity=0.5)
        self.place_in_area(parallelogram, 'B3', 'D5', scale_factor=0.65)
        self.play(Create(parallelogram))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF4500")
        vec_sum = Arrow(vec_a.get_start(), vec_a.get_end() + vec_b.get_vector(), color="#FF4500", buff=0)
        label_sum = MathTex("R", color="#FF4500")
        self.play(Create(vec_sum), Write(label_sum))
        
        # Apply critic fix: label positioning
        self.place_at_grid(label_a, 'D5', scale_factor=0.7)
        self.place_at_grid(label_b, 'B4', scale_factor=0.7)
        self.place_at_grid(label_sum, 'A5', scale_factor=0.7)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#808080")
        self.play(boat.animate.move_to(vec_sum.get_end()))
        self.wait(2)
