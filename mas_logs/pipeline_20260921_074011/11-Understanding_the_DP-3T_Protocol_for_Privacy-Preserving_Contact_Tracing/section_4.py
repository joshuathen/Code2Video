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
        self.setup_layout("Infection: The Disclosure Process", [
            "Positive users upload Daily Keys anonymously.", 
            "Keys are published to a public board.", 
            "Other phones download this keys list."
        ])
        
        # Elements
        user = Circle(radius=0.3, color=BLUE).set_fill(BLUE, opacity=0.5)
        key = Star(inner_radius=0.1, outer_radius=0.2, color="#00FFFF").set_fill("#00FFFF", opacity=1)
        
        # Asset integration
        server = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg").set_color("#FFFF00")
        phone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg").set_color("#FFFFFF")
        
        board = VGroup(*[Text("Key", font_size=16) for _ in range(3)]).arrange(DOWN)
        
        # Apply critic fixes for positioning
        self.place_at_grid(user, "B2", scale_factor=0.8) # Fix: Coder4-31
        self.place_at_grid(server, "B5", scale_factor=0.8)
        self.place_at_grid(phone, "D2", scale_factor=0.8) # Fix: Coder4-32
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(FadeIn(user), FadeIn(key))
        self.play(key.animate.move_to(self.grid["B5"]))
        self.play(FadeOut(key))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(board, "C5", scale_factor=0.8) # Fix: Coder4-30
        self.play(Create(board))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        copy_board = board.copy()
        self.play(copy_board.animate.move_to(self.grid["D2"]))
        self.wait(1)
