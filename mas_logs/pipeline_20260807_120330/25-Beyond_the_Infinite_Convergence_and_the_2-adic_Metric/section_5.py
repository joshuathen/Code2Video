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
        self.setup_layout("Convergence Criteria & Application", [
            "Convergence requires only that terms vanish.",
            "The 2-adic compass guides modern number theory.",
            "These metrics are vital for cryptography and beyond."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show the formula '|x+y| <= max(|x|, |y|)' (#FFD700) in a central box.
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        formula = MathTex(r"|x+y| \le \max(|x|, |y|)", color="#FFD700")
        box = SurroundingRectangle(formula, color="#FFD700", buff=0.2)
        formula_group = VGroup(box, formula)
        # Issue 28: Fix overlap, place formula_group in A2-B4
        self.place_in_area(formula_group, "A2", "B4", scale_factor=0.8)
        
        self.play(Create(box), Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate a '2-adic Compass' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg] 
        # needle (#FFFFFF) pointing towards a '0' icon.
        # Flash the numbers '2, 4, 8, 16' as the needle points more strongly toward '0'.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFFFF")
        )
        
        # Asset: compass.svg
        compass_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=WHITE)
        self.place_in_area(compass_asset, "D3", "E4", scale_factor=0.8)
        compass_center = compass_asset.get_center()
        
        # Issue 30: zero_icon at C3, scale 0.6
        zero_icon = Text("0", color=WHITE)
        self.place_at_grid(zero_icon, "C3", scale_factor=0.6)
        
        # Needle (Arrow)
        needle = Arrow(
            start=compass_center,
            end=compass_center + RIGHT * 0.6,
            buff=0,
            color=WHITE,
            stroke_width=6
        )
        
        self.play(FadeIn(compass_asset), Write(zero_icon), Create(needle))
        
        # Flash numbers and point needle
        numbers_vals = ["2", "4", "8", "16"]
        
        # Calculate target angle towards '0' icon (C3)
        zero_pos = self.grid["C3"]
        direction_vec = zero_pos - compass_center
        target_angle = np.arctan2(direction_vec[1], direction_vec[0])
        start_angle = needle.get_angle()

        for i, val in enumerate(numbers_vals):
            num_obj = Text(val, color=YELLOW).scale(0.6)
            # Flash positions around the compass
            flash_positions = ["D5", "E5", "E2", "D2"]
            self.place_at_grid(num_obj, flash_positions[i])
            
            # Rotate needle closer to '0'
            progress = (i + 1) / len(numbers_vals)
            current_target = start_angle + (target_angle - start_angle) * progress
            
            self.play(
                FadeIn(num_obj, scale=1.5),
                Rotate(needle, angle=current_target - needle.get_angle(), about_point=compass_center),
                run_time=0.5
            )
            self.play(FadeOut(num_obj), run_time=0.2)
            
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display labels 'Number Theory' and 'Cryptography' 
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg] (#8888FF) in the corners.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#8888FF")
        )
        
        label1 = Text("Number Theory", color="#8888FF")
        label2 = Text("Cryptography", color="#8888FF")
        
        # Asset: lock.svg
        lock_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg", color="#8888FF")
        
        # Positioning labels
        # label1 at A5-B6 (as per context check in issue 28)
        self.place_in_area(label1, "A5", "B6", scale_factor=0.6)
        
        # Issue 29: label2 (Cryptography) at F3-F4
        self.place_in_area(label2, "F3", "F4", scale_factor=0.6)
        
        # Place lock icon near cryptography label
        self.place_at_grid(lock_icon, "F5", scale_factor=0.4)
        
        self.play(FadeIn(label1), FadeIn(label2), FadeIn(lock_icon))
        self.wait(2)

        # Final fade out
        self.play(
            FadeOut(self.title),
            FadeOut(self.lecture),
            FadeOut(formula_group),
            FadeOut(compass_asset),
            FadeOut(zero_icon),
            FadeOut(needle),
            FadeOut(label1),
            FadeOut(label2),
            FadeOut(lock_icon)
        )
        self.wait(1)
