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
        self.setup_layout("The Finale: Substituting π", [
            "Let us set x to pi.", 
            "The cosine is negative one.", 
            "The sine is zero.", 
            "This gives e to the pi i.", 
            "It equals negative one exactly."
        ])
        
        # Prepare objects
        circle = Circle(radius=1, color=WHITE)
        self.place_in_area(circle, 'B3', 'E5', scale_factor=0.6)
        dot = Dot(point=circle.point_at_angle(0), color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(circle), FadeIn(dot))
        self.play(Rotate(dot, angle=PI, about_point=circle.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        math1 = MathTex(r"\cos(\pi) = -1")
        self.place_at_grid(math1, 'A5', scale_factor=0.9)
        self.play(Write(math1))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        math2 = MathTex(r"\sin(\pi) = 0").next_to(math1, DOWN)
        self.play(Write(math2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        euler = MathTex(r"e^{\pi i} = \cos(\pi) + i\sin(\pi)")
        self.place_in_area(euler, 'D2', 'D5', scale_factor=0.8)
        self.play(ReplacementTransform(VGroup(math1, math2), euler))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF0000")
        result = MathTex(r"e^{\pi i} + 1 = 0").scale(1.5).set_color(YELLOW)
        self.place_at_grid(result, 'D4')
        self.play(FadeOut(euler), FadeIn(result))
        self.play(Indicate(result))
        self.wait(2)
