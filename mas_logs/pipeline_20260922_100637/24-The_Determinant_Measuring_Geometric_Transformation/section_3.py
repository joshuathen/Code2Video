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
        lecture_lines = ["The sign shows the orientation of space.", "Positive determinant means original orientation is preserved.", "Negative means the space has been mirrored."]
        self.setup_layout("The Sign Significance: Orientation", lecture_lines)
        
        # Define mobjects
        arrow_preserved = Arrow(start=ORIGIN, end=RIGHT, color="#FF0000")
        arrow_flipped = Arrow(start=ORIGIN, end=LEFT, color="#FF0000")
        mirror_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.place_at_grid(arrow_preserved, 'C4', scale_factor=0.8)
        self.play(Create(arrow_preserved))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        det_label = Text("det > 0", color=WHITE, font_size=24)
        self.place_at_grid(det_label, 'B4', scale_factor=0.9)
        self.play(Write(det_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(Transform(arrow_preserved, arrow_flipped))
        
        self.place_at_grid(mirror_icon, 'D4', scale_factor=0.5)
        self.play(FadeIn(mirror_icon))
        
        det_label_neg = Text("det < 0", color=WHITE, font_size=24)
        self.place_at_grid(det_label_neg, 'E4', scale_factor=0.9)
        self.play(Write(det_label_neg))
        
        self.play(FadeOut(det_label))
        
        self.wait(2)
