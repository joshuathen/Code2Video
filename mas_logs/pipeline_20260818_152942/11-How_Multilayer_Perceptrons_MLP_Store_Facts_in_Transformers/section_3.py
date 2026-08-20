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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mechanism: Key-Value Memory Networks", [
            "Each neuron acts as a specific template.",
            "The first layer acts as a key.",
            "The second layer acts as a value."
        ])
        
        # Assets
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        
        # Animations based on storyboard:
        # Display Key matrix 'K' and Value matrix 'V'
        k_matrix = VGroup(Text("K", font_size=24).next_to(Matrix([["k_1"], ["k_2"]]), UP), Matrix([["k_1"], ["k_2"]])).set_color("#FFFFFF")
        v_matrix = VGroup(Text("V", font_size=24).next_to(Matrix([["v_1"], ["v_2"]]), UP), Matrix([["v_1"], ["v_2"]])).set_color("#FFFFFF")
        
        # Add assets
        icon1 = SVGMobject(asset_path).scale(0.5).set_color(WHITE)
        icon2 = SVGMobject(asset_path).scale(0.5).set_color(WHITE)
        
        self.place_at_grid(k_matrix, 'B3', scale_factor=0.6)
        self.place_at_grid(v_matrix, 'B5', scale_factor=0.6)
        self.place_at_grid(icon1, 'A3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(k_matrix), FadeIn(v_matrix), FadeIn(icon1))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Flash 'K' when comparing input query
        self.play(Indicate(k_matrix), run_time=1.5)
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Show weighted sum 'Σ v_i', output vector 'o', path from K to V
        sigma_v = MathTex(r"\\sum v_i").set_color("#00FFFF")
        out_vec = MathTex(r"o").set_color("#00FF00")
        arrow = Arrow(k_matrix.get_right(), v_matrix.get_left(), color=GREEN)
        
        self.place_at_grid(arrow, 'C4', scale_factor=0.7)
        self.place_at_grid(sigma_v, 'D4', scale_factor=0.7)
        self.place_at_grid(out_vec, 'E4', scale_factor=0.7)
        self.place_at_grid(icon2, 'E5', scale_factor=0.5)
        
        self.play(Create(arrow))
        self.play(FadeIn(sigma_v), FadeIn(out_vec), FadeIn(icon2))
        self.lecture[2].set_color(YELLOW)
        
        self.wait(2)
