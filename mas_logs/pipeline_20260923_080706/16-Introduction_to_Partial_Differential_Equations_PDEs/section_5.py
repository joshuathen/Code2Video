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
        self.setup_layout("Synthesis and Summary", [
            "PDEs are the language of physics.",
            "They allow predicting complex physical systems.",
            "Operators link math to real-world change."
        ])
        
        # Visuals
        summary_box = RoundedRectangle(corner_radius=0.1, height=4, width=5, color=BLUE).set_fill(BLUE, opacity=0.1)
        self.place_in_area(summary_box, 'A3', 'C6', scale_factor=0.8)
        
        # Assets
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        planet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg")
        
        # Labels for summary table
        labels = VGroup(
            Text("Weather", font_size=20),
            Text("Aerodynamics", font_size=20),
            Text("Structures", font_size=20),
            bridge
        ).arrange(DOWN, buff=0.3)
        self.place_in_area(labels, 'A4', 'C5', scale_factor=0.75)
        
        unifying_eq = VGroup(
            MathTex(r"\nabla^2 \phi = f", font_size=40, color=YELLOW),
            planet
        ).arrange(DOWN, buff=0.2)
        self.place_at_grid(unifying_eq, 'E4', scale_factor=1.0)
        
        self.play(FadeIn(summary_box), FadeIn(labels))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#32CD32"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#32CD32"))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#32CD32"), Write(unifying_eq))
        
        self.wait(2)
