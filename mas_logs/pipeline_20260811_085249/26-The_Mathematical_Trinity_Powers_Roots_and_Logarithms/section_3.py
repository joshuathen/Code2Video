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
        self.setup_layout("The Explorer's Tool: Logarithms", [
            "Logarithms find the time exponent.",
            "Think of a treasure map.",
            "Log_b(y) = x finds steps.",
            "10^3 = 1000 means log_10(1000) = 3.",
            "You need 3 rocket stages."
        ])
        
        # === Animation for Lecture Line 1 ===
        log_text = Text("log", font_size=48, color="#FFD700")
        self.place_at_grid(log_text, 'B2')
        self.play(FadeIn(log_text))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        self.place_at_grid(map_icon, 'D2', scale_factor=0.5)
        markers = VGroup(*[Text(str(i), color="#87CEEB").scale(0.5) for i in [1, 2, 3]])
        self.place_at_grid(markers, 'D4')
        self.play(FadeIn(map_icon), Write(markers))
        self.lecture[1].set_color("#87CEEB")

        # === Animation for Lecture Line 3 ===
        log_eq = MathTex(r"\log_b(y) = x", color="#FF4500")
        self.place_at_grid(log_eq, 'B4')
        self.play(FadeIn(log_eq))
        self.lecture[2].set_color("#FF4500")

        # === Animation for Lecture Line 4 ===
        conv_eq = MathTex(r"10^3 = 1000 \implies \log_{10}(1000) = 3", color="#32CD32")
        self.place_at_grid(conv_eq, 'E3')
        self.play(FadeIn(conv_eq))
        self.lecture[3].set_color("#32CD32")

        # === Animation for Lecture Line 5 ===
        rockets = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg").scale(0.3) for _ in range(3)])
        rockets.arrange(UP, buff=0.1)
        self.place_at_grid(rockets, 'F5')
        self.play(FadeIn(rockets))
        self.lecture[4].set_color("#FFFFFF")
        
        self.wait(2)
