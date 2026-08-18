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
        self.setup_layout("Why This Matters: Real-World Power", [
            "CLT enables precise statistical inference.",
            "We can use it for hypothesis testing.",
            "It builds a bridge to reliable data."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show CLT enabling inference on non-normal populations
        dist_shape = VGroup(
            Dot(color=BLUE), Dot(color=BLUE), Dot(color=BLUE),
            Dot(color=BLUE), Dot(color=BLUE)
        ).arrange(RIGHT)
        label1 = Text("Non-normal Population", font_size=18, color=WHITE).scale(0.7)
        group1 = VGroup(dist_shape, label1).arrange(DOWN)
        # Fix for issue 33: Expand to A1-B6
        self.place_in_area(group1, "A1", "B6", scale_factor=0.9)
        
        self.play(FadeIn(group1))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Display icons of finance, biology, and quality control
        icons = VGroup(
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chart.svg").set_color("#00FF00"),
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg").set_color("#00FF00"),
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg").set_color("#00FF00")
        ).arrange(RIGHT, buff=0.8)
        
        # Fix for issue 34: Use area C1-D6
        self.place_in_area(icons, "C1", "D6", scale_factor=0.85)
        
        self.play(FadeIn(icons))
        self.lecture[1].set_color("#00FF00")
        
        # === Animation for Lecture Line 3 ===
        # Final summary: 'The CLT is statistics' heartbeat'
        heart_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stethoscope.svg").set_color("#FF0000").scale(0.5)
        summary_text = Text("The CLT is statistics' heartbeat", font_size=24, color="#FF0000").scale(0.8)
        summary = VGroup(summary_text, heart_icon).arrange(RIGHT, buff=0.3)
        
        # Fix for issue 35: Use area E1-F6
        self.place_in_area(summary, "E1", "F6", scale_factor=0.9)
        
        self.play(Write(summary))
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
