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
        self.setup_layout("Real-World Application: The Saccharimeter", [
            "Industry measures sugar purity optically.",
            "Saccharimeter verifies sweetness levels.",
            "Impurities alter expected light twisting."
        ])
        
        # --- Asset Setup ---
        # Note: Using SVGMobject for placeholders as files aren't in this environment
        sensor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg", color="#CCCCCC")
        liquid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/liquid.svg", color=BLUE_E)
        analyzer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/analyzer.svg", color="#CCCCCC")
        light_beam = Line(LEFT, RIGHT, color=YELLOW)
        
        sugar_label = Text("Sugar Solution", font_size=16)

        # === Animation for Lecture Line 1 ===
        line1 = self.lecture[0]
        line1.set_color("#88CCFF")
        
        saccharimeter_group = VGroup(sensor, analyzer).arrange(RIGHT, buff=1.0)
        self.place_in_area(saccharimeter_group, "D2", "F5", scale_factor=0.6)
        self.play(FadeIn(saccharimeter_group))
        
        # === Animation for Lecture Line 2 ===
        line2 = self.lecture[1]
        line2.set_color("#88FF88")
        
        self.place_at_grid(liquid, "D3", scale_factor=0.5)
        self.place_at_grid(sugar_label, "E4", scale_factor=0.5)
        self.play(FadeIn(liquid), Write(sugar_label))
        
        # === Animation for Lecture Line 3 ===
        line3 = self.lecture[2]
        line3.set_color("#FF8888")
        
        self.place_at_grid(light_beam, "E6", scale_factor=0.7)
        self.play(Rotate(analyzer, angle=PI/8, about_point=analyzer.get_center()))
        self.play(Create(light_beam))
        self.wait(2)
