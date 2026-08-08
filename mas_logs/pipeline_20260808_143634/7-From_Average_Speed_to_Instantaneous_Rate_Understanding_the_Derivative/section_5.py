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
        lecture_lines = [
            "The derivative acts like a magnifying glass.",
            "It measures change at a single moment.",
            "The cheetah's speed at two seconds is 24."
        ]
        self.setup_layout("Application & Wrap-up", lecture_lines)
        
        # Define objects
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 30, 10], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 6 * x**2, x_range=[0, 3])
        tangent_line = Line(start=axes.c2p(1.5, 13.5), end=axes.c2p(2.5, 30.5), color="#32CD32")
        point_at_2 = Dot(axes.c2p(2, 24), color=YELLOW)
        
        # Asset: Cheetah
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        
        label = Text("Instantaneous Velocity", font_size=20, color=WHITE)
        speed_label = Text("24 m/s", font_size=24, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#32CD32"))
        self.place_in_area(axes, "A2", "D5", scale_factor=0.4)
        cheetah.scale(0.5).move_to(axes.c2p(2, 24))
        self.play(Create(axes), Create(curve), FadeIn(cheetah))
        self.play(Create(tangent_line))
        self.play(FadeIn(point_at_2))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_at_grid(label, "E4", scale_factor=0.6)
        self.play(Write(label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.place_at_grid(speed_label, "F4", scale_factor=0.7)
        self.play(Write(speed_label))
        
        # Final fade
        self.play(FadeOut(axes), FadeOut(curve), FadeOut(tangent_line), FadeOut(point_at_2), FadeOut(label), FadeOut(speed_label))
        cheetah_final = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        self.place_at_grid(cheetah_final, "C3", scale_factor=2.0)
        self.play(FadeIn(cheetah_final))
        
        self.wait(2)
