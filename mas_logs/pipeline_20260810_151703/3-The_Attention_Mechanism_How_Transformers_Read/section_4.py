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
            "All words attend to each other simultaneously.",
            "The model links related words directly.",
            "This resolves long-range dependencies efficiently."
        ]
        self.setup_layout("Self-Attention in Action", lecture_lines)
        
        # Define visual elements
        words = ["The", "cat", "chased", "the", "mouse", "it"]
        colors = [RED, GREEN, BLUE, YELLOW, PURPLE, WHITE]
        
        # Create mobjects
        word_mobjects = VGroup(*[Text(word, font_size=24, color=c) for word, c in zip(words, colors)])
        word_mobjects.arrange(RIGHT, buff=0.3)
        self.place_in_area(word_mobjects, 'B1', 'B6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(RED))
        
        # Load asset
        brain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg")
        
        # Visualize parallel heads
        head1 = VGroup(*[Circle(radius=0.15, color="#FF0000", fill_opacity=0.5) for _ in range(6)])
        head2 = VGroup(*[Circle(radius=0.15, color="#00FF00", fill_opacity=0.5) for _ in range(6)])
        head3 = VGroup(*[Circle(radius=0.15, color="#0000FF", fill_opacity=0.5) for _ in range(6)])
        
        heads = VGroup(brain, head1, head2, head3).arrange(DOWN, buff=0.5)
        self.place_in_area(heads, 'C1', 'E6', scale_factor=0.7)
        
        self.play(FadeIn(heads))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # Link 'it' (word_mobjects[5]) to 'cat' (word_mobjects[1])
        link = Line(word_mobjects[5].get_bottom(), word_mobjects[1].get_bottom(), color=WHITE)
        self.play(Create(link))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        # Merge vectors using network asset
        network = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/network.svg")
        final_vec = VGroup(network, Rectangle(width=2, height=0.5, color="#FFD700", fill_opacity=0.7)).arrange(DOWN)
        self.place_at_grid(final_vec, 'F3', scale_factor=0.8)
        
        self.play(
            FadeOut(heads),
            FadeOut(link),
            FadeIn(final_vec)
        )
        self.wait(1)
