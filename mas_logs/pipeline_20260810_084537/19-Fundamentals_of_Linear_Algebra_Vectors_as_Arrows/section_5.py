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
        self.setup_layout("Summary and Conclusion", [
            "Vectors represent motion; addition combines these paths.",
            "Scalars scale motion; basis vectors define our coordinates.",
            "Think of GPS: adding vectors for precise navigation."
        ])
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF6F61"))
        text1 = MathTex(r"v + w = \text{resultant path}", color=WHITE)
        self.place_in_area(text1, 'A4', 'A6', scale_factor=0.6)
        self.play(FadeIn(text1))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE),
                  self.lecture[1].animate.set_color("#6B5B95"))
        text2 = MathTex(r"c \cdot v = \text{scaled motion}", color=WHITE)
        self.place_in_area(text2, 'B4', 'B6', scale_factor=0.6)
        self.play(FadeIn(text2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE),
                  self.lecture[2].animate.set_color("#88B04B"))
        
        gps_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gps.svg", color="#88B04B")
        self.place_at_grid(gps_icon, 'C4', scale_factor=0.7)
        gps_label = Text("GPS Target", font_size=18).next_to(gps_icon, DOWN)
        
        self.play(FadeIn(gps_icon), Write(gps_label))

        self.wait(2)
        self.play(FadeOut(text1), FadeOut(text2), FadeOut(gps_icon), FadeOut(gps_label), FadeOut(self.lecture), FadeOut(self.title))
