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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Geometric Intuition: The Right-Hand Rule", [
            "The right-hand rule determines direction.",
            "Order of input vectors matters.",
            "The cross product is normal to the plane."
        ])
        
        # Load SVG assets
        hand = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")
        
        # Vectors: A, B, and Cross Product C
        vec_a = Vector(UP, color=WHITE)
        vec_b = Vector(RIGHT, color=WHITE)
        vec_c = Vector(OUT, color=WHITE)
        
        # Labels
        label_a = MathTex(r"\\vec{a}", font_size=24).next_to(vec_a.get_end(), UP, buff=0.1)
        label_b = MathTex(r"\\vec{b}", font_size=24).next_to(vec_b.get_end(), RIGHT, buff=0.1)
        label_c = MathTex(r"\\vec{a} \\times \\vec{b}", font_size=24).next_to(vec_c.get_end(), UP, buff=0.1)
        
        # Group them
        vectors = VGroup(vec_a, vec_b, vec_c, label_a, label_b, label_c)
        # Improvement: Using the requested area for better spacing
        self.place_in_area(vectors, 'C3', 'D5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        hand_copy = hand.copy().set_color("#FFFFFF")
        self.place_at_grid(hand_copy, 'D3', scale_factor=0.5)
        self.play(FadeIn(hand_copy), Create(vec_a), Create(vec_b), Write(label_a), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Flip vectors to show order matters
        self.play(
            vec_a.animate.rotate(PI, axis=RIGHT),
            label_a.animate.next_to(vec_a.get_end(), DOWN, buff=0.1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        hand_copy2 = hand.copy().set_color("#00FFFF")
        self.place_at_grid(hand_copy2, 'E3', scale_factor=0.5)
        self.play(FadeIn(hand_copy2), Create(vec_c), Write(label_c))
        self.wait(2)
