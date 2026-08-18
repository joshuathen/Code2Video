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
            "High-dimensional space is spiky and empty.",
            "Data points behave like isolated stars.",
            "Empty space challenges machine learning algorithms."
        ]
        self.setup_layout("Summary & Real-World Application", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Use Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg]
        self.lecture[0].set_color("#FFFFFF")
        
        # KEY CONCEPTS - Apply VideoCritic 34: Move to A4-C6
        key_concepts_raw = VGroup(
            Text("Scaling", font_size=24, color=WHITE),
            Text("Concentration", font_size=24, color=WHITE),
            Text("Sparseness", font_size=24, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        star_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg", color=WHITE)
        self.place_at_grid(star_icon, 'B5', scale_factor=0.3)
        
        key_concepts = VGroup(key_concepts_raw, star_icon)
        self.place_in_area(key_concepts, 'A4', 'C6', scale_factor=0.65)
        self.play(Write(key_concepts_raw), FadeIn(star_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Apply VideoCritic 33: Move stars to D3-F6
        self.lecture[1].set_color("#00FF00")
        stars = VGroup(*[Dot(point=np.random.uniform(-1, 1, 3) * 0.5, color="#00FF00") for _ in range(20)])
        self.place_in_area(stars, 'D3', 'F6', scale_factor=0.6)
        self.play(FadeIn(stars))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Use Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg]
        self.lecture[2].set_color("#FF00FF")
        algo_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg", color="#FF00FF")
        self.place_at_grid(algo_icon, 'E2', scale_factor=0.5)
        self.play(FadeIn(algo_icon), stars.animate.set_color("#FF00FF"))
        self.wait(2)
