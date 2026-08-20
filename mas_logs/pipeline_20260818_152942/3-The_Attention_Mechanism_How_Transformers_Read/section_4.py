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
        self.setup_layout("Synthesis: Putting it All Together", [
            "Multiply attention weights by values.", 
            "Create context-aware word vectors.", 
            "'It' absorbs 'animal' context."
        ])
        
        # Mobjects
        q_box = Square(side_length=0.8, color="#FF9999")
        k_box = Square(side_length=0.8, color="#99FF99")
        v_box = Square(side_length=0.8, color="#9999FF")
        
        q_label = Text("Q", font_size=20)
        k_label = Text("K", font_size=20)
        v_label = Text("V", font_size=20)
        
        q_group = VGroup(q_box, q_label)
        k_group = VGroup(k_box, k_label)
        v_group = VGroup(v_box, v_label)
        
        # Load asset
        animal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/animal.svg")
        
        # Grouped with more padding as requested
        central_block = VGroup(q_group, k_group, v_group, animal_icon).arrange(RIGHT, buff=0.4)
        
        # Apply layout fixes
        self.place_in_area(central_block, 'B3', 'C4', scale_factor=0.7)
        
        mha_label = Text("Multi-Head Attention", font_size=30, color="#FFFFFF")
        self.place_at_grid(mha_label, 'A4', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF9999"))
        self.play(FadeIn(q_group), FadeIn(k_group), FadeIn(v_group), FadeIn(animal_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#9999FF"))
        self.play(Write(mha_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#99FF99"))
        self.play(Flash(central_block, color="#FFFFFF", flash_radius=1.5, num_lines=12))
        self.wait(2)
