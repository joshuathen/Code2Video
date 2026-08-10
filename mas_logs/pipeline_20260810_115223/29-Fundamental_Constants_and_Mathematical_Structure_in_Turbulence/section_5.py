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
            "Turbulence structure governs engineering challenges.",
            "Aerodynamics and weather follow self-similar patterns.",
            "Nature exhibits strict mathematical order."
        ])
        
        # Elements
        recap_label = Text("Energy Cascade", font_size=32, color="#F1C40F")
        cloud_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg")
        cascade_group = VGroup(recap_label, cloud_icon).arrange(DOWN)
        
        airplane_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg")
        universal_text = Text("Turbulence is a universal process", font_size=28, color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#F1C40F"))
        self.place_at_grid(cascade_group, 'A3', scale_factor=0.9)
        self.play(FadeIn(cascade_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#F1C40F"))
        self.place_in_area(cloud_icon, 'B3', 'C4', scale_factor=0.5)
        self.play(FadeIn(cloud_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#F1C40F"))
        self.place_at_grid(universal_text, 'D3', scale_factor=0.8)
        self.place_at_grid(airplane_icon, 'E4', scale_factor=0.6)
        self.play(FadeIn(universal_text), FadeIn(airplane_icon))
        self.play(airplane_icon.animate.shift(RIGHT*1.5), run_time=2)
        self.wait(2)
