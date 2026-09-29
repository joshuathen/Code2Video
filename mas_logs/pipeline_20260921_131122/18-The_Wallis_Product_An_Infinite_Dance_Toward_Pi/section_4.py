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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Why does this involve only integers?",
            "Infinite products connect to transcendental pi.",
            "Each fraction steps along the circle."
        ]
        self.setup_layout("Intuition: Why Pi?", lecture_lines)
        
        # Load SVG Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        circle = Circle(radius=0.8, color=BLUE)
        pi_text = MathTex(r"\\pi", font_size=48, color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_color(YELLOW)
        
        self.place_at_grid(compass, 'C3', scale_factor=0.5)
        self.play(Create(compass))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[1].set_color(RED)
        
        self.place_at_grid(protractor, 'D4', scale_factor=0.5)
        self.place_at_grid(pi_text, 'D4', scale_factor=1.0)
        pi_text.next_to(protractor, RIGHT)
        self.play(Create(protractor), Write(pi_text))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[2].set_color(GREEN)
        
        self.place_at_grid(ruler, 'D3', scale_factor=0.5)
        self.place_at_grid(circle, 'D3', scale_factor=1.0)
        self.play(Create(ruler), Create(circle))
