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
        self.setup_layout("Conclusion and Impact", [
            "This remains mathematics' most elusive, unsolved prize.",
            "It holds the key to prime number distribution.",
            "The quest for the proof continues today."
        ])
        
        # Animation Elements
        prime_group = VGroup(
            Text("2", color=BLUE), Text("3", color=BLUE), Text("5", color=BLUE),
            Text("7", color=BLUE), Text("11", color=BLUE)
        ).arrange(RIGHT, buff=0.3)
        zeta_eq = MathTex(r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}", color=WHITE)
        
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg")
        
        lock = Text("🔒", font_size=72)
        library_bg = Rectangle(width=4, height=4, color=GRAY, fill_opacity=0.2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.place_in_area(library_bg, 'B2', 'E5')
        self.play(FadeIn(library_bg))
        self.place_at_grid(lock, 'C2')
        self.play(FadeIn(lock))
        self.place_at_grid(computer_icon, 'C5', scale_factor=0.5)
        self.play(FadeIn(computer_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.place_in_area(zeta_eq, 'B3', 'B5', scale_factor=0.9)
        self.place_at_grid(prime_group, 'D3', scale_factor=0.7)
        self.play(Write(zeta_eq), FadeIn(prime_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.place_at_grid(server_icon, 'E3', scale_factor=0.5)
        self.play(FadeIn(server_icon))
        question_mark = Text("?", font_size=96, color="#FFD700")
        self.place_at_grid(question_mark, 'C4')
        self.play(FadeIn(question_mark), run_time=2)
        self.play(Indicate(lock))
