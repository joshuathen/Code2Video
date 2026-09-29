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
        self.setup_layout("Putting it Together: The Final Proof Architecture", [
            "Structure your proof clearly and logically.",
            "Recognize hidden patterns to simplify integrals.",
            "Effective storytelling is key to proof architecture."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Draw proof architecture map [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg]
        proof_map = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg", color="#DCDCDC")
        self.place_in_area(proof_map, 'D1', 'F3', scale_factor=0.6)
        self.play(Create(proof_map))
        self.lecture[0].set_color("#DCDCDC")

        # === Animation for Lecture Line 2 ===
        # Flash connections between logic steps
        animation_path = Arrow(self.grid['C3'], self.grid['D5'], color="#32CD32")
        self.place_at_grid(animation_path, 'D4', scale_factor=0.7)
        self.play(Create(animation_path))
        self.lecture[1].set_color("#32CD32")

        # === Animation for Lecture Line 3 ===
        # Final result pops out prominently
        result_star = Star(color="#FFD700", fill_opacity=1)
        self.place_at_grid(result_star, 'E6', scale_factor=0.5)
        self.play(GrowFromCenter(result_star))
        self.lecture[2].set_color("#FFD700")

        self.wait(2)
