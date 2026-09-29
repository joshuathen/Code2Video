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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "A prisoner must guess a hidden coin's state.",
            "The board contains sixty-four coins, randomly placed.",
            "Can one flip guarantee the prisoner's survival?"
        ]
        self.setup_layout("The Hook: A High-Stakes Game", lecture_lines)
        
        # Assets
        board = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/board.svg", color="#E0FFFF")
        self.place_at_grid(board, 'D3', scale_factor=0.9)
        
        player = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prisoner.svg")
        player.set_stroke(color="#FFFFE0", width=4)
        self.place_at_grid(player, 'E5', scale_factor=0.8)
        
        choice_menu = VGroup(
            Text("Heads", font_size=24, color="#90EE90"),
            Text("Tails", font_size=24, color="#90EE90")
        ).arrange(RIGHT, buff=1.0)
        self.place_at_grid(choice_menu, 'B5', scale_factor=0.7)
        
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#E0FFFF"))
        self.play(DrawBorderThenFill(board))
        self.play(FadeIn(player))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E0FFFF"))
        self.play(FadeIn(choice_menu))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#E0FFFF"))
        path = Line(player.get_center(), choice_menu.get_center(), color="#FF8C00")
        self.play(Create(path))
        self.play(Flash(choice_menu, color=WHITE, flash_radius=0.5))
        
        self.place_at_grid(coin, 'C3', scale_factor=0.5)
        self.play(FadeIn(coin))
        self.wait(2)
