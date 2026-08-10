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
        lecture_lines = [
            "Vectors represent both magnitude and direction.",
            "Add two vectors: tip to tail.",
            "Scale vectors: stretching or shrinking length.",
            "Vector v (2,1) moves East 2, North 1.",
            "Vectors are building blocks of linear space."
        ]
        self.setup_layout("Prerequisite Refresh: Vectors as Arrows", lecture_lines)
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        
        # === Animation for Lecture Line 1 ===
        # Create vector v from origin (0,0) to point (2,1) using #FFFF00.
        v_vec = Arrow(start=ORIGIN, end=np.array([2, 1, 0]), color="#FFFF00", buff=0)
        v_label = Text("v", font_size=24, color=WHITE).next_to(v_vec.get_end(), UP, buff=0.1)
        v_group = VGroup(v_vec, v_label, compass.copy())
        
        self.place_at_grid(v_group, 'C4', scale_factor=0.6)
        self.play(Create(v_vec), Write(v_label), FadeIn(compass))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF5733")
        # Scale v by 1.5
        v_vec_scaled = Arrow(start=ORIGIN, end=np.array([3, 1.5, 0]), color="#FFFF00", buff=0)
        self.play(Transform(v_vec, v_vec_scaled))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF5733")
        u_vec = Arrow(start=v_vec.get_end(), end=np.array([4, 3, 0]), color="#0000FF", buff=0)
        self.play(Create(u_vec))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Final display of vector sum result as a new color #00FF00.
        self.lecture[4].set_color("#00FF00")
        sum_vec = Arrow(start=ORIGIN, end=np.array([4, 3, 0]), color="#00FF00", buff=0)
        self.play(ReplacementTransform(VGroup(v_vec, u_vec), sum_vec))
        self.play(FadeIn(compass.copy().move_to(self.grid['F1'])))
        self.wait(2)
