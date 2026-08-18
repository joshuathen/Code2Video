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
        self.setup_layout("The Solution Strategy: Parity and Symmetry", 
                          ["Parity and symmetry define our strategy.", 
                           "We track square configurations along the loop.", 
                           "Constraints vanish, forcing the square into existence."])
        
        # Define elements
        f_st = MathTex(r"f(s, \theta)", color="#FF6347")
        anti_sym = Text("Anti-Symmetry", color="#32CD32")
        borsuk = Text("Borsuk-Ulam Theorem", color="#FFD700")
        
        # Load asset as SVG (using a placeholder shape if file load fails in env)
        square_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color="#FF6347")
        inscribed_square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color="#00FFFF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(square_icon, 'B4', scale_factor=0.6)
        self.play(Write(f_st))
        self.place_at_grid(f_st, 'B4', scale_factor=0.9)
        self.play(FadeIn(square_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#32CD32"))
        self.place_at_grid(anti_sym, 'C4', scale_factor=0.8)
        self.play(FadeIn(anti_sym))
        self.place_at_grid(borsuk, 'D4', scale_factor=0.8)
        self.play(FadeIn(borsuk))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        group = VGroup(inscribed_square)
        self.place_in_area(group, 'E4', 'F6', scale_factor=0.9)
        self.play(Create(group))
        self.wait(2)
