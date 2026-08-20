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
        lecture_lines = ["Input passes through the MLP layer.", "Weights act as a retrieval mechanism.", "The layer adds a learned vector."]
        self.setup_layout("Case Study: Fact Updating", lecture_lines)
        
        # Mobjects for animation
        # Load assets
        keyboard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/keyboard.svg", color=GOLD)
        database_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/database.svg", color=WHITE)
        
        input_token = Text("Dog", color=BLUE)
        mlp_block = Rectangle(width=2, height=1.5, color=GREY, fill_opacity=0.3)
        weight_key = Text("Weight Key", font_size=20, color=YELLOW)
        feature_vec = Arrow(start=ORIGIN, end=RIGHT*1.5, color=GREEN)
        output_feat = Text("Barks", font_size=20, color=GREEN)

        self.place_at_grid(keyboard_icon, "B1", scale_factor=0.5)
        self.place_at_grid(input_token, "B1", scale_factor=0.8) # Positioned near keyboard
        
        self.place_at_grid(mlp_block, "B2", scale_factor=0.8)
        self.place_at_grid(weight_key, "B4", scale_factor=0.8)
        self.place_at_grid(database_icon, "D4", scale_factor=0.5)
        
        self.place_at_grid(feature_vec, "D3", scale_factor=0.8)
        self.place_at_grid(output_feat, "C4", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(input_token.animate.move_to(self.grid["B2"]))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Indicate(weight_key), Indicate(database_icon))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.play(GrowArrow(feature_vec))
        self.play(FadeIn(output_feat))
        self.wait(2)
