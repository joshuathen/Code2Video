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
        lecture_lines = ["Transformations map 3D space.", "Crucial for rendering computer graphics.", "Used in physics and animation."]
        self.setup_layout("Summary & Real-world Application", lecture_lines)

        # Helper to highlight lecture line
        def highlight_line(index, color=YELLOW):
            self.lecture[index].set_color(color)

        # Asset path
        bird_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bird.svg"
        bird = SVGMobject(bird_svg, color=BLUE)

        # === Animation for Lecture Line 1 ===
        highlight_line(0, BLUE)
        # Using recommendation from critic #35 for balance
        self.place_in_area(bird, 'D4', 'F6', scale_factor=0.6)
        self.play(FadeIn(bird))

        # === Animation for Lecture Line 2 ===
        highlight_line(0, WHITE)
        highlight_line(1, GREEN)
        
        # 'Squash and stretch' effect
        squashed = bird.copy().stretch(0.5, dim=1).stretch(1.5, dim=0)
        self.play(Transform(bird, squashed, run_time=1.5))

        # === Animation for Lecture Line 3 ===
        highlight_line(1, WHITE)
        highlight_line(2, RED)
        
        # Project onto a 2D line (screen)
        screen = Line(start=self.grid['E2'], end=self.grid['E5'], color=WHITE)
        self.play(Create(screen))
        
        # Projecting shadow
        shadow = bird.copy().stretch(0.01, dim=1).shift(DOWN*1.2)
        shadow.set_color(GRAY)
        self.play(Transform(bird, shadow), run_time=1.5)
        self.wait(1)
