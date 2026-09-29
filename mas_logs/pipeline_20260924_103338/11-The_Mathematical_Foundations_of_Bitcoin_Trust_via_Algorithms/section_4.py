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
        self.setup_layout("Security: Elliptic Curve Cryptography", [
            "Keys function as a one-way trapdoor.",
            "Sign messages using your secret private key.",
            "Use public keys to verify digital signatures."
        ])
        
        # Elliptic curve
        curve = ImplicitFunction(lambda x, y: y**2 - (x**3 - 2*x + 2), color="#00FFFF")
        self.place_in_area(curve, 'C4', 'F6', scale_factor=0.5)
        
        # Load Assets
        key_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg", color=YELLOW)
        lock_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg", color=GREEN)
        
        # Formula labels group
        G_label = Text("G", font_size=20)
        P_label = Text("P=kG", font_size=20)
        formula_labels = VGroup(G_label, P_label).arrange(RIGHT)
        
        # Annotations
        arc = Arc(start_angle=0, angle=TAU/8, radius=0.5, color=RED)
        theta = MathTex(r"\\theta", font_size=24, color=RED)
        annotations = VGroup(arc, theta, key_icon, lock_icon)

        # === Animation for Lecture Line 1 ===
        self.play(Create(curve), run_time=1)
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(formula_labels, 'B5', scale_factor=0.6)
        self.play(Write(formula_labels))
        self.place_at_grid(key_icon, 'C5', scale_factor=0.4)
        self.play(FadeIn(key_icon))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.place_in_area(annotations, 'E1', 'F6', scale_factor=0.4)
        self.play(FadeIn(annotations))
        self.place_at_grid(lock_icon, 'E5', scale_factor=0.4)
        self.play(FadeIn(lock_icon))
        self.lecture[2].set_color(GREEN)
        
        self.wait(2)
