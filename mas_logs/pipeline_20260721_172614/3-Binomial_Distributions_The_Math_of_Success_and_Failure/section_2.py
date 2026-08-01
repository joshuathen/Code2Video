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
        title = "Defining the Binomial Distribution"
        lecture_lines = [
            "In a binomial distribution, we repeat several trials.",
            "We count the probability of exactly k successes.",
            "Tree diagrams show how possible outcomes branch out."
        ]
        self.setup_layout(title, lecture_lines)
        
        color1 = "#33FF57"
        color2 = "#33C4FF"
        color3 = "#FF5733"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color1))
        
        # Create 5 slots using the provided asset
        slot_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/slot.svg"
        slots = VGroup(*[SVGMobject(slot_asset) for _ in range(5)]).set_color(WHITE)
        slots.arrange(RIGHT, buff=0.2)
        self.place_in_area(slots, 'B2', 'B5', scale_factor=0.8)
        
        n_formula = MathTex("n=5", color=color1)
        # Fix for Issue 25: Reposition n_formula
        self.place_at_grid(n_formula, 'A6', scale_factor=0.6)
        
        self.play(FadeIn(slots), Write(n_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color2))
        
        success_color = "#00FF00"
        failure_color = "#FF0000"

        success_labels = VGroup()
        failure_labels = VGroup()

        # Highlight 3 slots as 'Success'
        success_indices = [0, 2, 4]
        for i in success_indices:
            slots[i].set_fill(success_color, opacity=0.5)
            label = Text("S", font_size=24, color=WHITE).move_to(slots[i].get_center())
            success_labels.add(label)

        # Highlight 2 slots as 'Failure'
        failure_indices = [1, 3]
        for i in failure_indices:
            slots[i].set_fill(failure_color, opacity=0.5)
            label = Text("F", font_size=24, color=WHITE).move_to(slots[i].get_center())
            failure_labels.add(label)
        
        k_formula = MathTex("k=3", color=color2)
        # Fix for Issue 26: Reposition k_formula
        self.place_at_grid(k_formula, 'C3', scale_factor=0.7)

        self.play(
            LaggedStart(
                *[FadeIn(slots[i]) for i in success_indices],
                *[FadeIn(slots[i]) for i in failure_indices],
                lag_ratio=0.2
            ),
            Write(success_labels),
            Write(failure_labels),
            Write(k_formula)
        )
        self.wait(1)
        
        # Prepare for next animation
        self.play(FadeOut(slots), FadeOut(n_formula), FadeOut(k_formula), FadeOut(success_labels), FadeOut(failure_labels))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color3))
        
        # Create tree diagram components
        tree_diagram = VGroup()
        
        root_pos = np.array([0, 0, 0])
        root = Dot(root_pos, color=WHITE)
        root_label = Text("Start", font_size=20).next_to(root, UP)
        tree_diagram.add(root, root_label)
        
        animations = [FadeIn(root), Write(root_label)]
        
        # Function to create levels recursively to build the VGroup
        def add_level(parent_pos, level):
            if level > 3:
                return

            level_anims = []
            
            # Success branch
            s_pos = parent_pos + np.array([1, 1.5 / (1.5**level), 0])
            s_dot = Dot(s_pos, color=success_color)
            s_line = Line(parent_pos, s_pos, color=WHITE)
            s_label = MathTex("S", color=success_color).next_to(s_line.get_center(), UP, buff=0.1)
            tree_diagram.add(s_dot, s_line, s_label)
            level_anims.extend([Create(s_line), FadeIn(s_dot), Write(s_label)])
            
            # Failure branch
            f_pos = parent_pos + np.array([1, -1.5 / (1.5**level), 0])
            f_dot = Dot(f_pos, color=failure_color)
            f_line = Line(parent_pos, f_pos, color=WHITE)
            f_label = MathTex("F", color=failure_color).next_to(f_line.get_center(), DOWN, buff=0.1)
            tree_diagram.add(f_dot, f_line, f_label)
            level_anims.extend([Create(f_line), FadeIn(f_dot), Write(f_label)])

            return level_anims, s_pos, f_pos
        
        level1_anims, l1_s_pos, l1_f_pos = add_level(root_pos, 1)
        level2_s_anims, l2_ss_pos, l2_sf_pos = add_level(l1_s_pos, 2)
        level2_f_anims, l2_fs_pos, l2_ff_pos = add_level(l1_f_pos, 2)
        level3_ss_anims, _, _ = add_level(l2_ss_pos, 3)
        level3_sf_anims, _, _ = add_level(l2_sf_pos, 3)
        level3_fs_anims, _, _ = add_level(l2_fs_pos, 3)
        level3_ff_anims, _, _ = add_level(l2_ff_pos, 3)
        
        # Fix for Issue 27: Position the entire tree diagram in the specified area
        self.place_in_area(tree_diagram, 'B1', 'E6', scale_factor=0.8)

        self.play(*animations)
        self.play(*level1_anims)
        self.play(*level2_s_anims, *level2_f_anims)
        self.play(*level3_ss_anims, *level3_sf_anims, *level3_fs_anims, *level3_ff_anims)

        self.wait(2)
