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
        self.setup_layout("Conclusion: The Limit of Optimal Strategy", [
            "Minimax minimizes the maximum possible remaining words.", 
            "Optimal guesses prune the possibilities tree.", 
            "Narrow your search to the correct answer."
        ])
        
        # --- Visual Setup ---
        # Funnel representation using Polygon
        funnel = Polygon(
            [-1.5, 2, 0], [1.5, 2, 0], [0.5, -2, 0], [-0.5, -2, 0], 
            color=BLUE, fill_opacity=0.3
        )
        self.place_in_area(funnel, 'A3', 'D4', scale_factor=0.6)
        
        # Particles to represent words
        particles = VGroup(*[Dot(color=YELLOW).scale(0.5) for _ in range(20)])
        particles.arrange_in_grid(4, 5, buff=0.1)
        self.place_in_area(particles, 'A4', 'B6', scale_factor=0.4)
        
        # --- Animation for Lecture Line 1 ---
        # Minimax minimizes the maximum possible remaining words.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(particles.animate.move_to(self.grid['D5']), run_time=2)
        
        # --- Animation for Lecture Line 2 ---
        # Optimal guesses prune the possibilities tree.
        self.play(self.lecture[1].animate.set_color(GREEN))
        filter_line = Line(start=self.grid['D3'], end=self.grid['D6'], color=RED)
        self.play(Create(filter_line))
        self.play(particles.animate.move_to(self.grid['E5']), run_time=2)
        
        # --- Animation for Lecture Line 3 ---
        # Narrow your search to the correct answer.
        self.play(self.lecture[2].animate.set_color(GOLD))
        final_dot = Dot(color=WHITE).scale(1.5)
        self.place_at_grid(final_dot, 'E5', scale_factor=0.8)
        self.play(FadeIn(final_dot))
        self.play(FadeOut(particles), FadeOut(filter_line))
