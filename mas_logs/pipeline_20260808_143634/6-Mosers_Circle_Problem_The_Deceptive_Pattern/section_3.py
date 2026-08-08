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
        lecture_lines = ["Following the pattern, expect thirty-two.", "Wait, six points give only thirty-one.", "The beautiful pattern has finally broken."]
        self.setup_layout("The Counter-Intuitive Twist", lecture_lines)
        
        # Colors for lecture lines
        c1, c2, c3 = "#FFD700", "#FF4500", "#00CED1"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))
        
        # Using Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        seq = VGroup(Text("1, 2, 4, 8, 16, ? ", font_size=32), circle_icon.copy().scale(0.3))
        seq.arrange(RIGHT)
        
        # Fix 24: Use place_in_area
        self.place_in_area(seq, 'A3', 'B5', scale_factor=0.9)
        self.play(Write(seq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))
        
        # Assets as per storyboard
        res_32 = VGroup(Text("Expect: 32", font_size=28, color="#FF0000"), SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg").scale(0.25).set_color("#FF0000"))
        res_32.arrange(RIGHT)
        
        res_31 = VGroup(Text("Actual: 31", font_size=28, color="#00FF00"), SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg").scale(0.25).set_color("#00FF00"))
        res_31.arrange(RIGHT)
        
        # Fix 25: Use place_at_grid
        self.place_at_grid(res_32, 'D3', scale_factor=0.8)
        self.place_at_grid(res_31, 'D4', scale_factor=0.8)
        
        self.play(FadeIn(res_32))
        self.wait(0.5)
        self.play(FadeIn(res_31))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(c3))
        
        # Fix 26: Use place_at_grid
        broken = Text("Pattern Broken!", font_size=40, color=c3)
        self.place_at_grid(broken, 'C4', scale_factor=1.0)
        
        self.play(Write(broken))
        self.wait(2)
